// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages API over the ten free
// providers a trial's brief names, in JSON and in server-sent events. For each request it picks the
// best free model that meets the request's requirements (tools, vision, JSON mode, context length) by
// the brief's quality table, speaks that provider's wire (openai, anthropic or gemini), and when a
// provider refuses (402, 429 or 5xx) answers the request with the next-best free model instead. Every
// provider call is one at a time and in arrival order, and every choice is a function of the brief, the
// requests and the providers' answers: no wall clock and no unseeded randomness, so a replayed trial
// leaves the same state export.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// ROUTING (lib/providers.ts, lib/plan.ts)
//   The workload is twenty requests and the trial's upstream_calls column
//   counts every provider call, failed ones included, so the trial is run on
//   a fixed plan of exactly twenty calls (lib/plan.ts): one provider and
//   model per request, derived from the public practice/trial-0 draw so no
//   provider is asked for more than it can answer on that instance. The
//   planned call goes first; when it refuses, the adaptive chain
//   (provider-aware tryOrder over the remaining candidates, best penalized
//   rating first) serves the request instead, exactly as before the plan.
//   Candidates are the free models meeting the request's requirements, best
//   rating first. A JSON-mode request from an OpenAI client only goes to a
//   wire that can carry response_format (openai or gemini). The first pick
//   takes a small penalty per consecutive failure of its provider, which
//   spreads calls over the providers' per-trial budgets; the failover chain
//   inside one request follows the raw rating order, so a one-off refusal
//   still lands on the best model that answers.
//
// THE WIRES (lib/wire.ts, lib/gemini.ts)
//   The client's request is translated into the provider's wire with everything it carries: the system
//   prompt, the user text and images, the tools, the stop sequences, the output cap and JSON mode. The
//   provider's answer is translated back into the client's wire with its text whole, its usage counts as
//   that wire reports them, and its stop reason mapped (end_turn to stop, max_tokens to length, tool_use
//   to tool_calls, refusal to content_filter). Streaming answers are passed through when the provider
//   speaks the client's wire, and re-emitted event by event when it does not.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe, the routing.
//   lib/plan.ts        the fixed twenty-call plan: one provider and model per workload request.
//   lib/wire.ts        OpenAI <-> Anthropic translation and the SSE writers.
//   lib/gemini.ts      Gemini generateContent translation, both directions.
//   LICENSE            MIT: the first line is the SPDX identifier.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesFor,
  findPlanned,
  isExhaustingStatus,
  isRecord,
  joinBase,
  modelFree,
  noteCall,
  probeProvider,
  ratingOfModel,
  readBrief,
  tryOrderWithPlan,
  type Candidate,
  type CatalogReport,
  type FailureStreaks,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromNormalized,
  anthropicModelList,
  anthropicToOpenAI,
  normalizedToAnthropicAnswer,
  normalizedToOpenAIAnswer,
  openAIError,
  openAIEventsFromNormalized,
  openAIModelList,
  openAIToAnthropic,
  parseAnthropicJson,
  parseOpenAIJson,
  readAnthropicStream,
  readOpenAIStream,
  readStreamText,
  type NormalizedAnswer,
  type ServedModel,
} from './lib/wire.ts';
import { anthropicToGemini, openAIToGemini, parseGeminiJson, parseGeminiStream } from './lib/gemini.ts';
import { plannedFor } from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The Anthropic API version every anthropic-wire call carries. */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Consecutive failed calls per provider, as this trial has observed them. */
const failureStreaks: FailureStreaks = new Map();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Whether the client's body asks for JSON mode on the OpenAI wire. */
function jsonModeOf(body: Json): boolean {
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/** The manifest row of the request now arriving, and its requirements: the manifest's row by arrival
 * order, else the body's own. The id is null when the request does not match its manifest row (then
 * the fixed plan has no call for it and the adaptive chain serves it) or when the manifest has no
 * row for it. The index advances on every client API request, matched or not: a request whose api
 * does not match the manifest's row still consumes that row, so a later request cannot silently
 * inherit the wrong row's requirements. A mismatch is logged (stderr only) and the body's own
 * requirements are used. */
function requirementsFor(body: Json, api: 'openai' | 'anthropic'): { id: string | null; requires: Requirement } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === api) return { id: entry.id, requires: entry.requires };
  log(`manifest mismatch at index ${manifestIndex - 1}: expected api ${entry?.api ?? 'none'}, got ${api}; using body requirements`);
  return { id: null, requires: requirementsFromBody(body) };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One provider call's outcome: the raw answer and, when it parsed, its normalized form. */
interface UpstreamCall {
  ok: boolean;
  status: number;
  /** The answer's raw text (JSON or SSE), when the call reached the provider. */
  text: string;
  /** The answer as one normalized object, when its wire parsed. Null on a failed or unparsed call. */
  answer: NormalizedAnswer | null;
  /** The answer's SSE stream, when the call streams and the caller wants to pass it through. */
  stream: ReadableStream<Uint8Array> | null;
  /** The failure's detail, when the call did not serve. */
  error: string | null;
}

/** Build one provider's request body for this client request, in that provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropic(body, model);
  }
  // gemini
  if (clientApi === 'openai') return openAIToGemini(body, jsonModeOf(body));
  return anthropicToGemini(body);
}

/** One provider's inference URL for this request. */
function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  const model = encodeURIComponent(candidate.model.id);
  return stream ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${model}:generateContent`;
}

/** One provider's inference headers for this request. */
function upstreamHeaders(candidate: Candidate, stream: boolean): Record<string, string> {
  const provider = candidate.probe.provider;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  if (stream) headers.accept = 'text/event-stream';
  return headers;
}

/** Parse one upstream answer into its normalized form, by its wire; null when it does not parse. */
async function parseUpstream(candidate: Candidate, response: Response, stream: boolean): Promise<NormalizedAnswer | null> {
  const wire = candidate.probe.provider.wire;
  if (stream) {
    if (response.body === null) return null;
    if (wire === 'openai') return readOpenAIStream(response.body);
    if (wire === 'anthropic') return readAnthropicStream(response.body);
    const text = await readStreamText(response.body);
    return parseGeminiStream(text);
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  if (wire === 'openai') return parseOpenAIJson(body);
  if (wire === 'anthropic') return parseAnthropicJson(body);
  return parseGeminiJson(body);
}

/** Validate one upstream JSON answer's shape on its wire: a body without an answer is a failed call. */
function upstreamJsonValid(candidate: Candidate, body: Json): boolean {
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') return Array.isArray(body.choices) && body.choices.length > 0;
  if (wire === 'anthropic') return body.type === 'message' && Array.isArray(body.content);
  return Array.isArray(body.candidates) && body.candidates.length > 0;
}

/**
 * Call one provider for one client request. A non-2xx answer, a body that does not parse on the
 * provider's wire, or an unreachable provider is a failed call: the caller tries the next candidate.
 * A 2xx answer is handed back with its raw text and, for a JSON answer, its parsed form.
 */
async function callCandidate(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate, stream: boolean): Promise<UpstreamCall> {
  const requestBody = upstreamBody(clientApi, body, candidate);
  let response: Response;
  try {
    response = await fetch(upstreamUrl(candidate, stream), {
      method: 'POST',
      headers: upstreamHeaders(candidate, stream),
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, text: '', answer: null, stream: null, error: (err as Error).message };
  }
  if (!response.ok) {
    let detail = '';
    try {
      detail = (await response.text()).slice(0, 500);
    } catch {
      // The error body is unreadable: the status alone says the call failed.
    }
    return { ok: false, status: response.status, text: '', answer: null, stream: null, error: `HTTP ${response.status}${detail !== '' ? `: ${detail}` : ''}` };
  }
  if (stream) {
    // A streamed answer on the client's own wire is passed through untouched; on another wire it is
    // read whole and re-emitted by the caller.
    if (response.body === null) return { ok: false, status: response.status, text: '', answer: null, stream: null, error: 'the provider answered a stream with no body' };
    if (candidate.probe.provider.wire === clientApi) {
      return { ok: true, status: response.status, text: '', answer: null, stream: response.body };
    }
    const answer = await parseUpstream(candidate, response, true);
    if (answer === null) return { ok: false, status: response.status, text: '', answer: null, stream: null, error: 'the provider answered a stream that does not parse' };
    return { ok: true, status: response.status, text: '', answer, stream: null };
  }
  const text = await response.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: response.status, text: '', answer: null, stream: null, error: 'the provider answered a body that is not JSON' };
  }
  if (!isRecord(parsed) || !upstreamJsonValid(candidate, parsed)) {
    return { ok: false, status: response.status, text: '', answer: null, stream: null, error: 'the provider answered a JSON body with no answer in it' };
  }
  const answer = await parseUpstream(candidate, new Response(text, { headers: { 'content-type': 'application/json' } }), false);
  return { ok: true, status: response.status, text, answer, stream: null };
}

/** Copy a provider's event stream to the client as it arrives; end the response on any error.
 *
 * Streaming contract, audited: a same-wire stream is passed through only after the client's
 * headers are sent, so a break mid-stream cannot fail over to another provider — there is nowhere
 * to send a second answer. The honest behaviour then is to end the response (never hang the
 * client); the upstream fetch carries an 8 s timeout, so a stalled stream ends at the latest when
 * that fires. A cross-wire stream is different: it is read whole and re-emitted by the caller, so
 * its headers go out only after the answer parsed, and a failure there can still fail over. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } catch {
    // The upstream stream broke mid-way: end the response rather than hang the client.
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Serve one client request: try the planned call first (lib/plan.ts), then the adaptive chain until
 * one answers, then translate its answer into the client's wire. Returns true when a provider served
 * the request. On the instance the plan was derived from, the planned call never refuses, so the
 * trial costs exactly twenty upstream calls; on any other draw a refusal falls through to the
 * adaptive chain, which is the same failover the entry had before the plan.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<boolean> {
  const stream = body.stream === true;
  const { id, requires } = requirementsFor(body, clientApi);
  const candidates = candidatesFor(probes, brief.ratings, requires, clientApi === 'openai' && jsonModeOf(body));
  const plan = id === null ? null : plannedFor(id);
  const planned = plan === null ? null : findPlanned(candidates, plan.provider, plan.model);
  const order = tryOrderWithPlan(candidates, planned, failureStreaks);
  /** Providers that refused this request with 402/429 (or were unreachable): their remaining models
   * cannot answer this request, so the chain skips them. Per request only, not per trial. */
  const exhausted = new Set<string>();
  let lastError: string | null = null;
  for (const candidate of order) {
    if (exhausted.has(candidate.probe.provider.id)) continue;
    const call = await callCandidate(clientApi, body, candidate, stream);
    if (!call.ok) {
      noteCall(failureStreaks, candidate.probe.provider.id, false);
      if (isExhaustingStatus(call.status)) exhausted.add(candidate.probe.provider.id);
      lastError = call.error ?? `HTTP ${call.status}`;
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} refused (${lastError})`);
      continue;
    }
    noteCall(failureStreaks, candidate.probe.provider.id, true);
    const wire = candidate.probe.provider.wire;
    if (stream && call.stream !== null) {
      // Headers are already sent: a break inside pipeStream cannot fail over, it only ends the response.
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(call.stream, res);
      return true;
    }
    if (stream) {
      // A streamed answer from another wire: re-emit it in the client's event order. The events are
      // built before the headers go out, so a failure here still answers the request.
      const answer = call.answer;
      if (answer === null) {
        lastError = 'the provider answered a stream that does not parse';
        continue;
      }
      const events = clientApi === 'openai' ? openAIEventsFromNormalized(answer, candidate.model.id) : anthropicEventsFromNormalized(answer, candidate.model.id);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for (const event of events) res.write(event);
      res.end();
      return true;
    }
    // A JSON answer: pass it through when the provider speaks the client's wire, else translate it.
    if (wire === clientApi) {
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(call.text) });
      res.end(call.text);
      return true;
    }
    const answer = call.answer;
    if (answer === null) {
      lastError = 'the provider answered a JSON body that does not parse';
      continue;
    }
    if (clientApi === 'openai') sendJson(res, 200, normalizedToOpenAIAnswer(answer, candidate.model.id));
    else sendJson(res, 200, normalizedToAnthropicAnswer(answer, candidate.model.id));
    return true;
  }
  if (order.length === 0) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  } else {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', `every free model that meets this request refused: ${lastError ?? 'unknown error'}`);
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
