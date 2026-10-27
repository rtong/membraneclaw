"""Synthetic local benchmark. Run after installing the wheel; no external API."""
import asyncio
import platform
import statistics
import tempfile
import time
from pathlib import Path
from test_streaming import ScriptedStream, oa_events
from membraneclaw_runtime.streaming import StreamAssembler
from test_streaming import frame
from membraneclaw_runtime.providers.openai import OpenAIAdapter
from membraneclaw_runtime.providers.types import ModelRequest, ContentBlock
from membraneclaw_runtime.gateway import ModelGateway
from membraneclaw_runtime.tracing import FileTraceStore

async def main():
    events=[{"type":"response.output_text.delta","delta":"x"*128,"sequence_number":i} for i in range(30)]
    end=oa_events()[-1];end["sequence_number"]=30;end["response"]["output"][0]["content"][0]["text"]="x"*3840;events.append(end)
    rows=[]
    for traced in (False,True):
        totals=[];firsts=[];gaps=[];sizes=[]
        for n in range(5):
            with tempfile.TemporaryDirectory() as directory:
                received=[]
                if traced:
                    a=OpenAIAdapter();r=a.initial_state("gpt-4.1-2025-04-14").route
                    g=await ModelGateway.start(FileTraceStore(directory),a,ScriptedStream(events),r)
                    async def consume(f):received.append(time.perf_counter())
                    start=time.perf_counter();await g.generate(ModelRequest(r,(ContentBlock.text_block("synthetic"),)),stream=True,on_frame=consume)
                else:
                    a=StreamAssembler("responses_v1");start=time.perf_counter()
                    for e in events:a.feed(frame(e));received.append(time.perf_counter())
                    a.finish()
                duration=time.perf_counter()-start;totals.append(duration)
                firsts.append(received[0]-start);gaps.extend(y-x for x,y in zip(received,received[1:]));sizes.append(sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file()))
        rows.append(f"| {'FileTraceStore' if traced else 'assembler only'} | {statistics.median(firsts)*1000:.3f} | {statistics.median(gaps)*1000:.3f} | {31/statistics.median(totals):.1f} | {max(sizes)} |")
    print(f"Environment: {platform.system()} {platform.machine()}, Python {platform.python_version()}; 5 synthetic runs, 31 frames/run, 3,840 output characters. No network or provider latency. Files include commit/checkpoint metadata.\n")
    print('| Mode | First frame ms (median) | Frame gap ms (median) | Frames/s | Bytes on disk |\n|---|---:|---:|---:|---:|')
    print('\n'.join(rows))
asyncio.run(main())
