import asyncio
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from membraneclaw_runtime.rsi.cli import final_spec,main,demo_spec
from membraneclaw_runtime.rsi.controller import Controller
from membraneclaw_runtime.rsi.contracts import Invalid
from membraneclaw_runtime.rsi.journal import Journal
from membraneclaw_runtime.rsi.report import summary,write_reports
from helpers import run,spec


class Reports(unittest.TestCase):
    def test_semantic_digest_is_stable_but_record_digest_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            first = run(Path(temp)/'a',spec(1));second = run(Path(temp)/'b',spec(1))
            self.assertEqual(first['semantic_digest'],second['semantic_digest'])
            self.assertNotEqual(first['record_digest'],second['record_digest'])

    def test_report_rebuild_and_completed_resume_are_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run';first = run(path,spec(1))
            with Journal(path) as j:
                write_reports(j)
                (path/'report.json').unlink();(path/'report.md').unlink()
                second = write_reports(j)
                self.assertEqual(first,second)
                record = j.latest_record()
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['resume','--run-dir',str(path)]),0)
            with Journal(path) as j:self.assertEqual(j.latest_record(),record)
            self.assertNotIn('PRIVATE_EVALUATOR_CANARY',(path/'report.json').read_text())

    def test_final_evaluation_is_independent_linked_and_charged(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'search';run(path,spec(1))
            with Journal(path) as parent:
                before = parent.latest_record()
                final = final_spec(parent,{'operations':2,'units':2,'wall_seconds':100})
            with Journal(Path(temp)/'final',spec=final) as j:
                asyncio.run(Controller(j).run());report = summary(j)
                self.assertEqual(report['status'],'completed')
                self.assertEqual(report['decisions'],[])
                self.assertEqual(report['final_evaluation'][0]['score'],1)
                self.assertEqual(report['ledger']['calls'],2)
                self.assertEqual(report['linked_total_ledger']['calls'],15)
            with Journal(path) as parent:self.assertEqual(parent.latest_record(),before)

    def test_final_evaluation_requires_completed_search(self):
        with tempfile.TemporaryDirectory() as temp,Journal(Path(temp)/'search',spec=spec()) as j:
            with self.assertRaises(Invalid):final_spec(j,{'operations':2,'units':2,'wall_seconds':100})

    def test_cli_demo_verify_inspect_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            path = str(Path(temp)/'run')
            self.assertEqual(main(['demo','--output',path]),0)
            self.assertEqual(main(['verify','--run-dir',path]),0)
            self.assertEqual(main(['inspect','--run-dir',path]),0)
            self.assertEqual(main(['demo','--output',path]),1)

    def test_exported_example_matches_package_and_runs_after_editing(self):
        from importlib.resources import files
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            examples = Path(temp)/'example'
            self.assertEqual(main(['init-example','--output',str(examples)]),0)
            packaged = Path(str(files('membraneclaw_runtime.rsi').joinpath('data')))
            expected = {p.relative_to(packaged) for p in packaged.rglob('*') if p.is_file()}
            actual = {p.relative_to(examples) for p in examples.rglob('*') if p.is_file()}
            self.assertEqual(actual,expected)
            self.assertFalse((examples/'trace').exists())
            for relative in actual:
                path = examples/relative
                self.assertEqual(path.read_bytes(),(packaged/relative).read_bytes())
                self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(examples.stat().st_mode & 0o777,0o700)
            config = examples/'offline_config.json'
            value = json.loads(config.read_text());value['max_rounds'] = 1
            config.write_text(json.dumps(value))
            self.assertEqual(main(['run','--config',str(config),'--output',str(Path(temp)/'run')]),0)
            self.assertEqual(main(['verify','--run-dir',str(Path(temp)/'run')]),0)

    def test_export_refuses_existing_directory_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            existing = Path(temp)/'existing';existing.mkdir()
            self.assertEqual(main(['init-example','--output',str(existing)]),1)
            self.assertEqual(list(existing.iterdir()),[])
            marker = existing/'marker';marker.write_bytes(b'keep\r\n')
            link = Path(temp)/'link';link.symlink_to(existing,target_is_directory=True)
            self.assertEqual(main(['init-example','--output',str(link)]),1)
            self.assertEqual(main(['init-example','--output',str(marker)]),1)
            self.assertEqual(marker.read_bytes(),b'keep\r\n')

    def test_budget_stop_resume_cannot_override_the_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run';run(path,spec(1,operations=1))
            with Journal(path) as j:before = j.latest_record()
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['resume','--run-dir',str(path)]),2)
            with Journal(path) as j:self.assertEqual(j.latest_record(),before)

    def test_decision_verification_recomputes_from_evaluation_evidence(self):
        from membraneclaw_runtime.rsi.report import verify
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run';run(path,spec(1))
            with Journal(path) as j:
                j.store.read_episode(j.ref)  # Terminal history must remain untouched.
                saved = copy.deepcopy(j.state)
                j._resolve = lambda:None
                j.state['decisions'][0]['mean_gain'] = 999
                with self.assertRaises(Invalid):verify(j)
                j.state = saved
