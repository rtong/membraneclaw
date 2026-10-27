"""Offline RSI commands. No component dispatch during inspect/verify/report."""
import argparse
import asyncio
import copy
import json
from importlib.resources import files
from pathlib import Path
from .config import load_config, check_current_source, verify_spec
from .contracts import Invalid, hashed
from .controller import Controller, BUDGET_STOPS
from .journal import Journal
from .report import summary, verify, write_reports


def demo_spec():
    return load_config(files('membraneclaw_runtime.rsi').joinpath('data/offline_config.json'))


def init_example(output):
    """Export packaged fixtures without dispatching components or overwriting files."""
    destination = Path(output)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)

    def copy_directory(source, target):
        for entry in sorted(source.iterdir(), key=lambda item: item.name):
            path = target / entry.name
            if entry.is_dir():
                path.mkdir(mode=0o700)
                copy_directory(entry, path)
            else:
                with path.open('xb') as stream:
                    path.chmod(0o600)
                    stream.write(entry.read_bytes())

    copy_directory(files('membraneclaw_runtime.rsi').joinpath('data'), destination)
    return {'output':str(destination.resolve()),
            'configuration':str((destination / 'offline_config.json').resolve()),
            'synthetic_only':True}


def final_spec(parent, limits):
    verify(parent)
    if parent.state['status'] != 'completed' or parent.spec['purpose'] != 'search':raise Invalid('a completed search is required')
    spec = copy.deepcopy(parent.spec)
    spec['purpose'] = 'final_test'
    spec['parent'] = {'record':parent.latest_record(),'selected':parent.state['incumbent'],
                      'versions':parent.state['versions'],'ledger':parent.state['ledger']}
    spec['config']['limits'] = limits
    spec['config_digest'] = hashed(spec['config'])
    spec['inputs']['source_files']['configuration'] = json.dumps(spec['config'],sort_keys=True)
    spec['dataset_digest'] = hashed(spec['inputs'])
    verify_spec(spec)
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    sub = subs.add_parser('init-example',help='export editable synthetic fixtures to a new directory')
    sub.add_argument('--output',required=True)
    for command in ('demo','run'):
        sub = subs.add_parser(command);sub.add_argument('--output',required=True)
        if command == 'run':sub.add_argument('--config',required=True)
    for command in ('resume','inspect','verify','report','final-evaluate'):
        sub = subs.add_parser(command);sub.add_argument('--run-dir',required=True)
        if command == 'resume':sub.add_argument('--recompute-offline',action='store_true')
        if command == 'final-evaluate':
            sub.add_argument('--output',required=True)
            sub.add_argument('--operations',type=int,required=True)
            sub.add_argument('--units',type=int,required=True)
            sub.add_argument('--wall-seconds',type=int,required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init-example':
            print(json.dumps(init_example(args.output),indent=2))
            return 0
        if args.command in {'demo','run'}:
            spec = demo_spec() if args.command == 'demo' else load_config(args.config)
            with Journal(args.output,spec=spec) as journal:
                asyncio.run(Controller(journal).run());result = write_reports(journal)
        elif args.command == 'final-evaluate':
            with Journal(args.run_dir) as parent:
                spec = final_spec(parent,{'operations':args.operations,'units':args.units,'wall_seconds':args.wall_seconds})
            with Journal(args.output,spec=spec) as journal:
                asyncio.run(Controller(journal).run());result = write_reports(journal)
        else:
            with Journal(args.run_dir,repair=args.command == 'resume') as journal:
                if args.command == 'resume' and journal.state['status'] != 'completed' and journal.state['stop_reason'] not in BUDGET_STOPS:
                    check_current_source(journal.spec);verify(journal);journal.prepare_resume()
                    asyncio.run(Controller(journal,recompute=args.recompute_offline).run())
                result = verify(journal) if args.command == 'verify' else write_reports(journal) if args.command == 'report' else summary(journal)
        print(json.dumps(result,indent=2))
        return 0 if args.command == 'verify' or result.get('status') == 'completed' else 2
    except Exception as error:
        print(json.dumps({'error':type(error).__name__,'detail':str(error)}))
        return 1


if __name__ == '__main__':raise SystemExit(main())
