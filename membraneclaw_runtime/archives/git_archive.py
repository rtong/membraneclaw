"""Append-only snapshots; fresh isolated checkout, ordinary push, remote verification."""
import fcntl
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from .base import ArchivePolicy, ArchiveReceipt
from ..providers.types import ConfigurationError
from ..tracing.schema import Busy, Corruption, canonical_json, digest


class ArchiveFailure(Exception):
    pass


class GitHubVisibility:
    """Verify repository metadata with an operator-provided read credential."""
    def __init__(self, credential_env='TRACE_ARCHIVE_GITHUB_TOKEN'):
        self.credential_env = credential_env

    def __call__(self, remote):
        match = re.fullmatch(r'(?:https://github\.com/|git@github\.com:)([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?', remote)
        if not match:
            raise ConfigurationError('GitHub visibility cannot be verified for this target')
        token = os.environ.get(self.credential_env)
        if not token:
            raise ConfigurationError('archive visibility verification credential missing')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        request = urllib.request.Request('https://api.github.com/repos/'+match[1]+'/'+match[2],
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
        try:
            with urllib.request.build_opener(NoRedirect).open(request,timeout=30) as response:
                metadata = json.load(response)
        except Exception:
            raise ConfigurationError('archive visibility verification failed') from None
        if type(metadata.get('private')) is not bool:
            raise ConfigurationError('archive visibility response invalid')
        return 'private' if metadata['private'] else 'public'


class GitTraceArchive:
    def __init__(self, *, remote=None, branch=None, state_dir='./var/trace-archive', visibility_check=None,
                 command_timeout=120, before_push=None):
        self.remote, self.branch = remote, branch
        self.state_dir = Path(state_dir).resolve()
        self.visibility_check = visibility_check or GitHubVisibility()
        self.command_timeout = command_timeout
        self.before_push = before_push  # Fault injection/test coordination; never part of trace data.
        if remote:
            if remote.startswith('-') or '\n' in remote or '\r' in remote:
                raise ConfigurationError('invalid archive remote')
            parsed = urlparse(remote)
            if parsed.scheme in {'http','https'} and (parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ConfigurationError('credential-bearing archive URL forbidden')
        if branch and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_/-]*', branch):
            raise ConfigurationError('invalid archive branch')

    def _git(self, checkout, *args, check=True):
        try:
            result = subprocess.run(['git',*args],cwd=checkout,capture_output=True,text=True,
                timeout=self.command_timeout,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
        except (OSError,subprocess.TimeoutExpired):
            raise ArchiveFailure('git_command_failed') from None
        if check and result.returncode:
            raise ArchiveFailure('git_command_failed')
        return result

    def _blob(self, checkout, revision, path):
        try:
            return read_git_blob(checkout, revision, path, timeout=self.command_timeout)
        except (Corruption, OSError, subprocess.TimeoutExpired):
            raise ArchiveFailure('archive_blob_verification_failed') from None

    def _receipt(self, receipt):
        self.state_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
        path = self.state_dir/(receipt.snapshot_hash+'.receipt.json')
        fd, temporary = tempfile.mkstemp(dir=self.state_dir,prefix='.receipt-')
        try:
            with os.fdopen(fd,'wb') as file:
                file.write(canonical_json(asdict(receipt)));file.flush();os.fsync(file.fileno())
            os.replace(temporary,path)
            descriptor = os.open(self.state_dir,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return receipt

    def publish(self, snapshot, policy=None):
        policy = policy or ArchivePolicy()
        snapshot.verify()
        if not self.remote or not self.branch:
            return ArchiveReceipt('not_configured',snapshot.snapshot_hash)
        if policy.visibility != 'private' or policy.data_mode != 'private_raw':
            raise ConfigurationError('raw snapshots require a private target')
        if type(policy.max_attempts) is not int or policy.max_attempts < 1:
            raise ConfigurationError('positive archive attempt limit required')
        if self.visibility_check(self.remote) != 'private':
            raise ConfigurationError('raw archive target is not private')
        self.state_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
        fd = os.open(self.state_dir/'worker.lock',os.O_WRONLY|os.O_CREAT,0o600)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise Busy('archive worker busy') from None
            for attempt in range(policy.max_attempts):
                try:
                    with tempfile.TemporaryDirectory(dir=self.state_dir,prefix='checkout-') as temp:
                        checkout = Path(temp)
                        self._git(checkout,'init','-q')
                        self._git(checkout,'config','core.hooksPath','/dev/null')
                        self._git(checkout,'remote','add','origin',self.remote)
                        heads = self._git(checkout,'ls-remote','--heads','origin','refs/heads/'+self.branch).stdout.strip()
                        if heads:
                            self._git(checkout,'fetch','-q','origin','refs/heads/'+self.branch)
                            self._git(checkout,'checkout','-q','-b','archive','FETCH_HEAD')
                        else:
                            self._git(checkout,'checkout','-q','-b','archive')
                        relative = Path('traces')/snapshot.episode.key/snapshot.snapshot_hash
                        target = checkout/relative
                        manifest = canonical_json({'snapshot_hash':snapshot.snapshot_hash,'episode':asdict(snapshot.episode),
                                                   'revision':snapshot.revision,'inventory':snapshot.inventory()})
                        files = dict(snapshot.files)
                        files['snapshot.json'] = manifest
                        already = target.exists()
                        if already:
                            if target.is_symlink() or any((target/p).is_symlink() or not (target/p).is_file() or self._blob(checkout,'HEAD',str(relative/p))!=b for p,b in files.items()):
                                raise Corruption('archive snapshot conflict')
                        else:
                            for name,data in files.items():
                                path = target/name
                                # A tracked symlink in an ancestor must not redirect private writes.
                                if any(parent.is_symlink() for parent in path.parents):
                                    raise Corruption('symlink in archive destination')
                                path.parent.mkdir(parents=True,exist_ok=True)
                                with path.open('xb') as file:
                                    os.chmod(path,0o600);file.write(data)
                            paths = [str(relative/p) for p in sorted(files)]
                            for path in paths:
                                oid = self._git(checkout,'hash-object','-w','--no-filters',path).stdout.strip()
                                self._git(checkout,'update-index','--add','--cacheinfo','100644',oid,path)
                            self._git(checkout,'-c','user.name=MembraneClaw Trace Archive','-c','user.email=trace-archive@localhost',
                                      '-c','commit.gpgsign=false','commit','-q','-m','Archive trace snapshot '+snapshot.snapshot_hash)
                        commit = self._git(checkout,'rev-parse','HEAD').stdout.strip()
                        if not already:
                            if self.before_push:self.before_push(checkout,attempt)
                            # Push errors/timeouts are ambiguous. The fresh verifier below is authoritative.
                            try:self._git(checkout,'push','origin','HEAD:refs/heads/'+self.branch)
                            except ArchiveFailure:pass
                        self._git(checkout,'fetch','-q','origin','refs/heads/'+self.branch)
                        remote_commit = self._git(checkout,'rev-parse','FETCH_HEAD').stdout.strip()
                        if self._git(checkout,'merge-base','--is-ancestor',commit,remote_commit,check=False).returncode:
                            raise ArchiveFailure('remote_advanced_or_push_failed')
                        for name,data in files.items():
                            if self._blob(checkout,remote_commit,str(relative/name)) != data:
                                raise ArchiveFailure('remote_snapshot_verification_failed')
                        return self._receipt(ArchiveReceipt('published',snapshot.snapshot_hash,self.branch,commit,repository=self.remote))
                except ArchiveFailure:
                    # Reapply only this immutable snapshot on the newest branch in a fresh checkout.
                    continue
            return self._receipt(ArchiveReceipt('pending',snapshot.snapshot_hash,self.branch,error_code='archive_retry_exhausted',repository=self.remote))
        finally:
            os.close(fd)


def read_git_blob(repository, revision, path, *, timeout=120):
    if not re.fullmatch(r'HEAD|[a-f0-9]{40,64}', revision):
        raise Corruption('unsafe archive revision')
    result = subprocess.run(['git','cat-file','blob',revision+':'+path],cwd=repository,
                            capture_output=True,timeout=timeout)
    if result.returncode:raise Corruption('archive blob unavailable')
    return result.stdout


def _assemble_snapshot(read):
    from ..tracing.schema import EpisodeRef
    from ..tracing.validation import SealedSnapshot
    metadata = json.loads(read('snapshot.json'))
    files=[]
    for item in metadata['inventory']['files']:
        name=item['path']
        if not re.fullmatch(r'genesis\.json|commits/[0-9]{6,}\.json|payloads/[a-f0-9]{64}\.bin',name):
            raise Corruption('unsafe archive inventory path')
        data=read(name)
        if len(data)!=item['size'] or digest(data)!=item['sha256']:raise Corruption('archive payload mismatch')
        files.append((name,data))
    snapshot=SealedSnapshot(EpisodeRef(**metadata['episode']),metadata['revision'],tuple(sorted(files)),metadata['snapshot_hash'])
    snapshot.verify()
    if snapshot.inventory()!=metadata['inventory']:raise Corruption('archive inventory mismatch')
    return snapshot


def load_git_snapshot(repository, snapshot_path):
    """Read committed bytes directly; supports bare/no-checkout repositories."""
    from pathlib import PurePosixPath
    path = PurePosixPath(snapshot_path)
    if path.is_absolute() or '..' in path.parts or str(path) != snapshot_path:
        raise Corruption('unsafe snapshot path')
    # Pin HEAD once, so concurrent ref updates cannot mix revisions.
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    return _assemble_snapshot(lambda name: read_git_blob(repository,revision,str(path/name)))


def load_snapshot(directory):
    """Read archive objects when inside Git, or validate a standalone snapshot."""
    directory = Path(directory)
    if directory.is_symlink():raise Corruption('symlink snapshot directory')
    directory = directory.resolve()
    result = subprocess.run(['git','rev-parse','--show-toplevel'],cwd=directory,capture_output=True,text=True)
    if result.returncode == 0:
        repository = Path(result.stdout.strip()).resolve()
        relative = directory.relative_to(repository).as_posix()
        tracked = subprocess.run(['git','cat-file','-e','HEAD:'+relative+'/snapshot.json'],
                                 cwd=repository,capture_output=True)
        if tracked.returncode == 0:
            return load_git_snapshot(repository,relative)
    def read(name):
        path=directory/name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise Corruption('symlink archive payload')
        return path.read_bytes()
    return _assemble_snapshot(read)
