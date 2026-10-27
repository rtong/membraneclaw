import asyncio
import json
import os
import socket
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from membraneclaw_runtime.gateway import ModelGateway
from membraneclaw_runtime.providers.openai import OpenAIAdapter
from membraneclaw_runtime.providers.anthropic import AnthropicAdapter
from membraneclaw_runtime.providers.gemini import GeminiAdapter
from membraneclaw_runtime.providers.types import ModelRequest,ContentBlock
from membraneclaw_runtime.runtime import AgentRuntime
from membraneclaw_runtime.tools import ToolRegistry,ToolSpec
from membraneclaw_runtime.tracing import FileTraceStore
from membraneclaw_runtime.tracing.schema import Corruption,canonical_json
from membraneclaw_runtime.tracing.export import export_bundle,restore_bundle,load_bundle
from membraneclaw_runtime.tracing.replay import execute_replay,playback,RecordedProvider,RecordedTool,ReplayMismatch
from helpers import FakeTransport

SECRET="private-prompt-and-opaque-signature"

def responses(provider):
    if provider=="openai":
        return [{"status":"completed","output":[{"type":"reasoning","encrypted_content":SECRET,"summary":[]},
            {"type":"function_call","call_id":"c1","name":"calc","arguments":'{"x":2}'}]},
            {"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}]
    if provider=="anthropic":
        return [{"type":"message","role":"assistant","content":[{"type":"thinking","thinking":"summary","signature":SECRET},
            {"type":"tool_use","id":"c1","name":"calc","input":{"x":2}}],"stop_reason":"tool_use"},
            {"type":"message","role":"assistant","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}]
    return [{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"calc","args":{"x":2}},"thoughtSignature":SECRET}]},"finishReason":"STOP"}]},
            {"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}]}]

class ReplayContracts(unittest.IsolatedAsyncioTestCase):
    async def recording(self,root,provider="openai"):
        a={"openai":OpenAIAdapter,"anthropic":AnthropicAdapter,"gemini":GeminiAdapter}[provider]()
        model={"openai":"gpt-4.1-2025-04-14","anthropic":"claude-sonnet-4-20250514","gemini":"gemini-2.5-flash"}[provider]
        store=FileTraceStore(root);calls=[]
        def tool(args):calls.append(args);return {"value":"x"*10000,"x":args["x"]}
        registry=ToolRegistry([ToolSpec("calc","synthetic",{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]},tool,"test_v1")])
        g=await ModelGateway.start(store,a,FakeTransport(responses(provider)),a.initial_state(model).route)
        req=ModelRequest(g.route,(ContentBlock.text_block(SECRET),),tools=registry.definitions(),max_output_tokens=2048)
        result=await AgentRuntime(g,registry,req).run();self.assertEqual(result.status,"completed")
        return store,g,registry,calls
    async def test_three_real_protocols_replay_without_network_or_real_tool(self):
        with tempfile.TemporaryDirectory() as root:
            for provider in ("openai","anthropic","gemini"):
                store,g,registry,calls=await self.recording(Path(root)/provider,provider)
                with patch.object(socket,"create_connection",side_effect=AssertionError("network forbidden")):
                    result=await execute_replay(store,g.recorder.episode)
                self.assertEqual(result["text"],"done");self.assertEqual(result["model_requests"],2);self.assertEqual(len(calls),1)
    async def test_private_export_restore_and_analysis_contains_no_private_payload(self):
        with tempfile.TemporaryDirectory() as root:
            store,g,_,_=await self.recording(Path(root)/"source")
            private=Path(root)/"private.zip";analysis=Path(root)/"analysis.zip"
            export_bundle(store,g.recorder.episode,private,mode="private-replay")
            export_bundle(store,g.recorder.episode,analysis)
            self.assertEqual(private.stat().st_mode&0o777,0o600)
            with zipfile.ZipFile(analysis) as z:
                combined=b''.join(z.read(p) for p in z.namelist());self.assertNotIn(SECRET.encode(),combined)
            with self.assertRaises(Corruption):load_bundle(analysis)
            restored,episode=restore_bundle(private,Path(root)/"restored")
            self.assertEqual(playback(restored,episode),playback(store,g.recorder.episode))
            self.assertEqual((await execute_replay(restored,episode))["status"],"completed")
    async def test_tampered_payload_and_unsafe_archive_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            store,g,_,_=await self.recording(Path(root)/"source")
            source=Path(root)/"p.zip";export_bundle(store,g.recorder.episode,source,mode="private-replay")
            target=Path(root)/"changed.zip"
            with zipfile.ZipFile(source) as src,zipfile.ZipFile(target,"w") as dest:
                for name in src.namelist():dest.writestr(name,src.read(name)+(b'changed' if name.endswith('.bin') else b''))
            with self.assertRaises(Corruption):load_bundle(target)
    async def test_request_and_tool_argument_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            store,g,registry,_=await self.recording(root)
            recorded=RecordedProvider(store,g.recorder.episode)
            native=g._last_output.native_request
            async def nothing(*args):pass
            with self.assertRaises(ReplayMismatch):await recorded.exchange(replace(native,body=b'{"changed":true}'),before_send=nothing,after_receive=nothing,timeout=1)
            tool=RecordedTool(store,g.recorder.episode)
            from membraneclaw_runtime.providers.types import ToolCall
            call=ToolCall("new_runtime_id","c1","calc",'{"x":999}',{"x":999},1)
            with self.assertRaises(ReplayMismatch):await tool.invoke(registry.prepare(call))
    async def test_cli_validate_and_export_are_reviewable(self):
        from membraneclaw_runtime.tracing.cli import main
        import contextlib,io
        with tempfile.TemporaryDirectory() as root:
            store,g,_,_=await self.recording(root)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code=main(["validate","--trace-dir",root,"--episode",g.recorder.episode.episode_id])
            self.assertEqual(code,0);self.assertEqual(json.loads(out.getvalue())["status"],"completed")
