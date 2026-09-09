import { createUIMessageStream, createUIMessageStreamResponse } from 'ai';
import type { ResearchUIMessage, TraceEvent } from '@/lib/types';

const backend = process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8100';

type SseBuffer = { remainder: string; dataLines: string[] };

function consumeSseChunk(buffer: SseBuffer, chunk: string): string[] {
  buffer.remainder += chunk;
  const lines = buffer.remainder.split(/\r?\n/);
  buffer.remainder = lines.pop() ?? '';
  const events: string[] = [];

  for (const line of lines) {
    if (line === '') {
      if (buffer.dataLines.length) {
        events.push(buffer.dataLines.join('\n'));
        buffer.dataLines = [];
      }
      continue;
    }
    if (line.startsWith(':')) continue;
    if (line.startsWith('data:')) buffer.dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  return events;
}

export async function POST(req: Request) {
  const body = await req.json();
  const messages = body.messages ?? [];
  const last = messages[messages.length - 1];
  const text = (last?.parts ?? []).filter((p:any)=>p.type==='text').map((p:any)=>p.text).join('\n').trim();
  const documentIds: string[] = body.documentIds ?? [];
  const targetDocumentId: string | undefined = body.targetDocumentId;
  const workspaceId = body.workspaceId ?? 'default';
  if (!text || !documentIds.length) return new Response('Question and at least one document are required', { status: 400 });

  const start = await fetch(`${backend}/api/runs`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({workspace_id:workspaceId,document_ids:documentIds,target_document_id:targetDocumentId,question:text}), signal:req.signal });
  if (!start.ok) return new Response(await start.text(), { status:start.status });
  const run = await start.json();

  const stream = createUIMessageStream<ResearchUIMessage>({
    async execute({ writer }) {
      writer.write({ type:'data-run', id:`run-${run.id}-start`, data:{runId:run.id,status:'running'} });
      const textId = `answer-${run.id}`; let textStarted = false;
      let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
      try {
        const res = await fetch(`${backend}/api/runs/${run.id}/events`, {
          headers:{Accept:'text/event-stream'}, signal:req.signal,
        });
        if (!res.ok || !res.body) throw new Error('Could not open agent trace stream');
        reader = res.body.getReader();
        const decoder = new TextDecoder();
        const sseBuffer: SseBuffer = { remainder:'', dataLines:[] };

        while (!req.signal.aborted) {
          const {done,value} = await reader.read();
          if (done) break;
          for (const raw of consumeSseChunk(sseBuffer, decoder.decode(value, {stream:true}))) {
            if (!raw || raw === '[DONE]') continue;
            try {
              const evt = JSON.parse(raw) as TraceEvent;
              writer.write({ type:'data-trace', id:evt.id, data:evt });
              if(evt.artifact_id) writer.write({ type:'data-artifact', id:`artifact-${evt.artifact_id}`, data:{id:evt.artifact_id,kind:evt.type,path:'',label:evt.title} });
              const answer = typeof evt.payload?.human_answer==='string' ? String(evt.payload.human_answer) : '';
              if(evt.type==='answer' && answer){
                if(!textStarted){ writer.write({type:'text-start',id:textId}); textStarted=true; }
                writer.write({type:'text-delta',id:textId,delta:answer});
                writer.write({type:'text-end',id:textId});
                writer.write({type:'data-answer',id:`answer-data-${run.id}`,data:{runId:run.id,answer}});
              }
            } catch (error) {
              console.error('Ignoring malformed backend SSE event', error);
            }
          }
        }
      } catch (error) {
        if (!req.signal.aborted) throw error;
      } finally {
        if (reader) {
          await reader.cancel().catch(() => undefined);
          reader.releaseLock();
        }
      }

      if (req.signal.aborted) return;
      const finalRes=await fetch(`${backend}/api/runs/${run.id}`,{cache:'no-store', signal:req.signal}); const finalRun=finalRes.ok?await finalRes.json():{status:'failed',error:'Could not load terminal run'};
      if(!textStarted){
        const fallback = finalRun.final_answer || `The research run ended with status ${finalRun.status}. ${finalRun.error??'Inspect the trace and failure record.'}`;
        writer.write({type:'text-start',id:textId}); writer.write({type:'text-delta',id:textId,delta:fallback}); writer.write({type:'text-end',id:textId});
      }
      writer.write({ type:'data-run', id:`run-${run.id}-end`, data:{runId:run.id,status:finalRun.status} });
    },
    onError: err => err instanceof Error ? err.message : 'Research stream failed',
  });
  return createUIMessageStreamResponse({ stream });
}
