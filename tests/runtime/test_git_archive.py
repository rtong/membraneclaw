import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from membraneclaw_runtime.archives.base import ArchivePolicy
from membraneclaw_runtime.archives.git_archive import GitTraceArchive, ArchiveFailure, load_snapshot
from membraneclaw_runtime.providers.types import ConfigurationError
from membraneclaw_runtime.tracing import FileTraceStore,TraceRecorder,Payload
from membraneclaw_runtime.tracing.schema import Event
from membraneclaw_runtime.tracing.validation import seal,restore,validate


def git(cwd,*args):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True,stderr=subprocess.DEVNULL).strip()

class GitArchiveContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.remote=self.root/'remote.git';git(self.root,'init','--bare',str(self.remote))
        self.store=FileTraceStore(self.root/'traces');self.rec=TraceRecorder.begin_episode(self.store,{'synthetic':True})
        payload=Payload.json({'synthetic':'full private tool result'})
        self.rec.append_event(Event('tool.completed',payload_ref=payload.ref(self.rec.episode)),payload=payload)
        self.rec.finish_episode('completed',capture='complete',external_outcome='known')
        self.snapshot=seal(self.store,self.rec.episode)
    def archive(self,**kwargs):
        return GitTraceArchive(remote=str(self.remote),branch='trace-data',state_dir=self.root/'worker',visibility_check=lambda remote:'private',**kwargs)
    def test_publish_verify_idempotence_and_restore(self):
        archive=self.archive();receipt=archive.publish(self.snapshot)
        self.assertEqual(receipt.status,'published')
        count=git(self.remote,'rev-list','--count','trace-data')
        repeated=archive.publish(self.snapshot);self.assertEqual(repeated.status,'published')
        self.assertEqual(git(self.remote,'rev-list','--count','trace-data'),count)
        checkout=self.root/'copy';git(self.root,'clone','--branch','trace-data',str(self.remote),str(checkout))
        relative=Path('traces')/self.snapshot.episode.key/self.snapshot.snapshot_hash
        for name,data in self.snapshot.files:self.assertEqual((checkout/relative/name).read_bytes(),data)
        restored=restore(load_snapshot(checkout/relative),self.root/'restored');self.assertEqual(validate(restored,self.snapshot.episode).status,'completed')
    def test_push_failure_retains_local_then_safe_retry(self):
        archive=self.archive();original=archive._git
        def fail(cwd,*args,**kwargs):
            if args[0]=='push':raise ArchiveFailure('synthetic_failure')
            return original(cwd,*args,**kwargs)
        with patch.object(archive,'_git',side_effect=fail):receipt=archive.publish(self.snapshot,ArchivePolicy(max_attempts=2))
        self.assertEqual(receipt.status,'pending');self.assertEqual(validate(self.store,self.rec.episode).status,'completed')
        self.assertEqual(archive.publish(self.snapshot).status,'published')
    def test_push_succeeded_but_reply_lost_does_not_duplicate(self):
        archive=self.archive();original=archive._git
        def uncertain(cwd,*args,**kwargs):
            result=original(cwd,*args,**kwargs)
            if args[0]=='push':raise ArchiveFailure('synthetic_timeout_after_success')
            return result
        with patch.object(archive,'_git',side_effect=uncertain):self.assertEqual(archive.publish(self.snapshot).status,'published')
        self.assertEqual(git(self.remote,'rev-list','--count','trace-data'),'1')
    def test_non_fast_forward_reapplies_without_overwriting_remote_files(self):
        # Seed target first with another episode, then race a third-party change before push.
        self.assertEqual(self.archive().publish(self.snapshot).status,'published')
        other=TraceRecorder.begin_episode(self.store,{'synthetic':'second'});other.finish_episode('completed')
        second=seal(self.store,other.episode)
        def advance(checkout,attempt):
            if attempt:return
            rival=self.root/'rival';git(self.root,'clone','--branch','trace-data',str(self.remote),str(rival))
            (rival/'unrelated.txt').write_text('must survive')
            git(rival,'add','--','unrelated.txt')
            git(rival,'-c','user.name=Synthetic','-c','user.email=synthetic@localhost','-c','commit.gpgsign=false','commit','-m','Unrelated remote update')
            git(rival,'push','origin','trace-data')
        self.assertEqual(self.archive(before_push=advance).publish(second).status,'published')
        self.assertEqual(git(self.remote,'show','trace-data:unrelated.txt'),'must survive')
        self.assertEqual(git(self.remote,'rev-list','--count','trace-data'),'3')
    def test_visibility_and_unconfigured_target_fail_closed(self):
        unconfigured=GitTraceArchive(state_dir=self.root/'unused')
        self.assertEqual(unconfigured.publish(self.snapshot).status,'not_configured')
        self.assertFalse((self.root/'unused').exists())
        with self.assertRaises(ConfigurationError):self.archive().publish(self.snapshot,ArchivePolicy(visibility='public'))
        archive=self.archive();archive.visibility_check=lambda remote:'public'
        with self.assertRaises(ConfigurationError):archive.publish(self.snapshot)
    def test_same_revision_seals_same_hash_and_does_not_include_later_changes(self):
        self.assertEqual(seal(self.store,self.rec.episode).snapshot_hash,self.snapshot.snapshot_hash)
        # The immutable snapshot contains fixed bytes even if an unrelated episode changes.
        other=TraceRecorder.begin_episode(self.store,{'synthetic':'later'});other.finish_episode('failed')
        self.assertEqual(self.archive().publish(self.snapshot).status,'published')
        self.assertEqual(self.snapshot.snapshot_hash,seal(self.store,self.rec.episode).snapshot_hash)

    def test_local_health_distinguishes_archive_and_trace(self):
        from membraneclaw_runtime.operations import health
        archive=self.archive();archive.publish(self.snapshot)
        gauges=health(self.store,archive_state=archive.state_dir)
        self.assertEqual(gauges['episodes'],1)
        self.assertEqual(gauges['archive_published'],1)
        self.assertEqual(gauges['archive_pending'],0)
        self.assertGreater(gauges['storage_bytes'],0)

    def test_exact_blobs_with_text_attributes_and_clean_smudge_filters(self):
        from membraneclaw_runtime.archives.git_archive import load_git_snapshot
        import os
        # Configure a real transforming filter inherited by the archive worker.
        config=self.root/'gitconfig'
        config.write_text('[filter "transform"]\n clean = "tr a-z A-Z"\n smudge = "tr a-z A-Z"\n required = true\n')
        seed=self.root/'seed';git(self.root,'init',str(seed))
        (seed/'.gitattributes').write_text('*.bin text eol=lf filter=transform\n')
        git(seed,'add','.gitattributes')
        git(seed,'-c','user.name=Synthetic','-c','user.email=synthetic@localhost','-c','commit.gpgsign=false','commit','-m','Attributes')
        git(seed,'push',str(self.remote),'HEAD:refs/heads/trace-data')
        rec=TraceRecorder.begin_episode(self.store,{'synthetic':'attributes'})
        payload=Payload(b'original\r\nbytes\r\n')
        rec.append_event(Event('tool.completed',payload_ref=payload.ref(rec.episode)),payload=payload)
        rec.finish_episode('completed',capture='complete',external_outcome='known')
        snapshot=seal(self.store,rec.episode)
        relative=Path('traces')/snapshot.episode.key/snapshot.snapshot_hash
        with patch.dict(os.environ,{'GIT_CONFIG_GLOBAL':str(config)}):
            archive=self.archive();self.assertEqual(archive.publish(snapshot).status,'published')
            self.assertEqual(archive.publish(snapshot).status,'published')
            for name,data in snapshot.files:
                actual=subprocess.check_output(['git','cat-file','blob','trace-data:'+str(relative/name)],cwd=self.remote)
                self.assertEqual(actual,data)
            checkout=self.root/'filtered';git(self.root,'clone','--branch','trace-data',str(self.remote),str(checkout))
            # Checkout transformations actually changed the working file.
            name=next(n for n,b in snapshot.files if b==payload.data)
            self.assertNotEqual((checkout/relative/name).read_bytes(),payload.data)
            self.assertEqual(load_snapshot(checkout/relative).files,snapshot.files)
            bare=self.root/'bare-copy';git(self.root,'clone','--bare','--branch','trace-data',str(self.remote),str(bare))
            restored=restore(load_git_snapshot(bare,relative.as_posix()),self.root/'exact-restored')
            self.assertEqual(validate(restored,snapshot.episode).status,'completed')
            self.assertEqual(restored.read_payload(payload.ref(snapshot.episode)),payload.data)

    def test_standalone_snapshot_inside_unrelated_git_checkout(self):
        checkout=self.root/'unrelated';git(self.root,'init',str(checkout))
        snapshot_dir=checkout/'download'
        for name,data in self.snapshot.files:
            path=snapshot_dir/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        from dataclasses import asdict
        from membraneclaw_runtime.tracing.schema import canonical_json
        (snapshot_dir/'snapshot.json').write_bytes(canonical_json({
            'snapshot_hash':self.snapshot.snapshot_hash,'episode':asdict(self.snapshot.episode),
            'revision':self.snapshot.revision,'inventory':self.snapshot.inventory()}))
        self.assertEqual(load_snapshot(snapshot_dir).files,self.snapshot.files)
