# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over the
best free AI models it can reach — no paid model, no API key of its own. It reads a trial brief
(`$MAKEFUN_PROMPT_FILE`) naming ten free-tier providers, probes each provider's catalog route in the
provider's own wire, and answers each client request with the highest-rated free model that meets the
request's requirements (tools, vision, JSON mode, context length). When a provider refuses
(402, 429 or 5xx) it fails over to the next-best free model instead of passing the error on.

## How it works

- **Routing** (`lib/providers.ts`): candidates are free models meeting the request's requirements,
  ordered by the brief's quality table. A JSON-mode request from an OpenAI client only goes to a wire
  that can carry `response_format` (OpenAI or Gemini). The first pick takes a small penalty per
  consecutive failure of its provider, which spreads calls over the providers' per-trial budgets; the
  failover chain inside one request follows the raw rating order.
- **Wires** (`lib/wire.ts`, `lib/gemini.ts`): the client's request is translated into the provider's
  wire with everything it carries — system prompt, user text and images, tools and tool results, stop
  sequences, output cap, JSON mode — and the provider's answer is translated back into the client's
  wire with its text whole, its usage counts as that wire reports them, and its stop reason mapped
  (`end_turn` → `stop`, `max_tokens` → `length`, `tool_use` → `tool_calls`, `refusal` →
  `content_filter`). Streaming answers pass through untouched when the provider speaks the client's
  wire, and are re-emitted event by event when it does not.
- **Trial loop** (`main.ts`): probe, build one catalog, listen on `0.0.0.0` at a free port, register
  the port and catalog with the env, serve `GET /v1/models`, `POST /v1/chat/completions` and
  `POST /v1/messages` (JSON and SSE), poll `GET /work` every 100 ms, and exit 0 when the env says
  the workload is done. `node main.ts --list` prints the same catalog document and exits 0.
- **Determinism**: no wall clock and no unseeded randomness anywhere, so a replayed trial leaves the
  same state export byte for byte.

## Run it

```sh
node main.ts --list        # print the catalog and exit (empty outside a trial)
```

Inside a trial the env sets `MAKEFUN_PROMPT_FILE`, `MAKEFUN_ENV_URL` and `MAKEFUN_TRIAL_SEED`, and
the server is started as `node main.ts` from this directory. Node 22+ with the standard library
alone; nothing to install.

## Test it locally

```sh
# From the unpacked kit directory, with the process driver (no Docker needed):
node cli/makefun.mjs grade-local --kit . --bot <this directory> --driver process
node cli/makefun.mjs grade-local --kit . --bot <this directory> --driver process --instances 3
```

That runs the practice trials (0–2) live against the simulated providers and then replays each from
its cassette, printing the score and every column. No provider API key is needed: this entry makes no
model calls of its own — its only calls are to the trial env's providers.

## License

MIT — see `LICENSE` (first line: `SPDX-License-Identifier: MIT`).
