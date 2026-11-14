// The client wires: OpenAI Chat Completions and Anthropic Messages, in both directions.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. Each provider also speaks one of
// those wires, so every request is a translation between the client's wire and the provider's wire, and
// every answer is translated back:
//
//   - openai -> openai: the body passes through with the routed model's id (and stream_options
//     include_usage added for a stream, so the provider's usage chunk reaches the client);
//   - anthropic -> openai: the Messages request is rewritten to Chat Completions (system prompt, image
//     blocks, tool definitions, tool results and tool choice included) and the answer is rewritten back
//     into a Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated;
//   - openai -> anthropic and anthropic -> anthropic: the same translations in the other direction.
//     The Gemini wire lives in lib/gemini.ts; its answers come back here as normalized answers.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** The normalized shape of one provider answer, whatever wire it arrived on. */
export interface NormalizedAnswer {
  /** The answer's text, whole, exactly as the provider wrote it. */
  text: string;
  /** Tool calls the provider made, in order (empty when the answer is plain text). */
  toolCalls: { id: string; name: string; arguments: string }[];
  /** The provider's stop reason, in its own wire's spelling. */
  stopReason: string | null;
  /** The provider's usage: input and output token counts, as that wire reports them. */
  usage: { input: number; output: number } | null;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

/** The four stop families every wire's stop reason falls into, in OpenAI's spelling. */
export type StopFamily = 'stop' | 'length' | 'tool_calls' | 'content_filter';

/** Any wire's stop reason as its family: done, hit the cap, called a tool, or was refused. */
export function stopFamilyOf(reason: unknown): StopFamily {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tool_calls';
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** Any wire's stop reason in OpenAI's spelling. */
export function openAIStop(reason: unknown): string {
  return stopFamilyOf(reason);
}

/** Any wire's stop reason in Anthropic's spelling. */
export function anthropicStop(reason: unknown): string {
  const family = stopFamilyOf(reason);
  if (family === 'length') return 'max_tokens';
  if (family === 'tool_calls') return 'tool_use';
  if (family === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Backwards-compatible alias: OpenAI's finish_reason as Anthropic's stop_reason. */
export const stopReasonOf = openAIStopToAnthropic;

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages become the system prompt, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through. response_format cannot be carried
 * on this wire and is dropped: JSON-mode requests are never routed here (see providers.ts).
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : '';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = typeof content === 'string' ? content : textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : [{ type: 'text', text: '' }] });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : toolResultText(content) }],
      });
      continue;
    }
    // A user message: string content, or parts of text and images.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) {
    const stops = body.stop.filter((s): s is string => typeof s === 'string');
    if (stops.length > 0) out.stop_sequences = stops;
  } else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropicAnswer(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content = body.content;
  const text = answerText(content);
  const blocks = Array.isArray(content) ? content : [];
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'call_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: anthropicStopToOpenAI(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A normalized answer as a Chat Completions object (for a gemini-wire provider's answer). */
export function normalizedToOpenAIAnswer(answer: NormalizedAnswer, model: string): Json {
  const toolCalls: Json[] = answer.toolCalls.map((call) => ({
    id: call.id,
    type: 'function',
    function: { name: call.name, arguments: call.arguments },
  }));
  const input = answer.usage?.input ?? 0;
  const output = answer.usage?.output ?? 0;
  return {
    id: 'chatcmpl_aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) },
        finish_reason: openAIStop(answer.stopReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A normalized answer as an Anthropic Messages object (for a gemini-wire provider's answer). */
export function normalizedToAnthropicAnswer(answer: NormalizedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsOf(call.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(answer.stopReason),
    stop_sequence: null,
    usage: { input_tokens: answer.usage?.input ?? 0, output_tokens: answer.usage?.output ?? 0 },
  };
}

/** The text of a message's content: a string, or its text blocks joined. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('');
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text into the pieces one SSE stream carries; '' alone when there is none. */
export function splitText(text: string, size = 13): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * A normalized answer as the OpenAI event stream: a role chunk, content deltas (and tool call deltas),
 * a finish chunk, a usage chunk, then [DONE]. The usage chunk is what lets the client check the
 * provider's token counts on a stream.
 */
export function openAIEventsFromNormalized(answer: NormalizedAnswer, model: string): string[] {
  const events: string[] = [];
  const chunk = (delta: Json, finish: string | null, usage?: Json): Json => ({
    id: 'chatcmpl_aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage !== undefined ? { usage } : {}),
  });
  events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) events.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, index) => {
    events.push(sseEvent(null, chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, null)));
    if (call.arguments !== '') {
      events.push(sseEvent(null, chunk({ tool_calls: [{ index, function: { arguments: call.arguments } }] }, null)));
    }
  });
  events.push(sseEvent(null, chunk({}, openAIStop(answer.stopReason))));
  const input = answer.usage?.input ?? 0;
  const output = answer.usage?.output ?? 0;
  events.push(sseEvent(null, { id: 'chatcmpl_aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } }));
  events.push('data: [DONE]\n\n');
  return events;
}

/**
 * A normalized answer as the Anthropic event stream: message_start, one content block per text or tool
 * call, message_delta with the stop reason and output tokens, then message_stop.
 */
export function anthropicEventsFromNormalized(answer: NormalizedAnswer, model: string): string[] {
  const events: string[] = [];
  const input = answer.usage?.input ?? 0;
  const output = answer.usage?.output ?? 0;
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } },
    }),
  );
  let index = 0;
  if (answer.text !== '') {
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } }));
    if (call.arguments !== '') {
      events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  events.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(answer.stopReason), stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openAIStopToAnthropic(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * An Anthropic SSE stream as the OpenAI event stream: a role chunk, text and tool call deltas, a finish
 * chunk, a usage chunk, then [DONE]. The whole upstream stream is read first, so the usage chunk can
 * carry the provider's exact token counts.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const answer = await readAnthropicStream(upstream);
  yield* openAIEventsFromNormalized(
    { text: answer.text, toolCalls: answer.toolCalls, stopReason: answer.stopReason, usage: answer.usage },
    model,
  );
}

/** Read a whole Anthropic SSE stream into one normalized answer (text, tool calls, stop, usage). */
export async function readAnthropicStream(upstream: ReadableStream<Uint8Array>): Promise<NormalizedAnswer> {
  const text = await readStreamText(upstream);
  return parseAnthropicStream(text);
}

/** Read a whole OpenAI SSE stream into one normalized answer (text, tool calls, stop, usage). */
export async function readOpenAIStream(upstream: ReadableStream<Uint8Array>): Promise<NormalizedAnswer> {
  const text = await readStreamText(upstream);
  return parseOpenAIStream(text);
}

/** One ReadableStream's bytes as text. */
export async function readStreamText(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let out = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      out += decoder.decode(value, { stream: true });
    }
    out += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return out;
}

/** The SSE frames of one event-stream text: each frame's event name and its data lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An Anthropic event-stream text as one normalized answer. */
export function parseAnthropicStream(text: string): NormalizedAnswer {
  let outText = '';
  let input = 0;
  let output = 0;
  let stopReason: string | null = null;
  const toolCalls: { id: string; name: string; arguments: string }[] = [];
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && typeof data.index === 'number' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const call = { id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${data.index}`, name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' };
      byIndex.set(data.index, call);
      toolCalls.push(call);
    } else if (frame.event === 'content_block_delta' && typeof data.index === 'number' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') outText += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const call = byIndex.get(data.index);
        if (call !== undefined) call.arguments += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: outText, toolCalls, stopReason, usage: { input, output } };
}

/** An OpenAI event-stream text as one normalized answer. */
export function parseOpenAIStream(text: string): NormalizedAnswer {
  let outText = '';
  let input = 0;
  let output = 0;
  let stopReason: string | null = null;
  const toolCalls: { id: string; name: string; arguments: string }[] = [];
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') outText += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : byIndex.size;
      let entry = byIndex.get(index);
      if (entry === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        entry = { id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${index}`, name: typeof fn.name === 'string' ? fn.name : '', arguments: '' };
        byIndex.set(index, entry);
        toolCalls.push(entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
    }
    if (typeof choice.finish_reason === 'string') stopReason = choice.finish_reason;
  }
  return { text: outText, toolCalls, stopReason, usage: { input, output } };
}

/** A Chat Completions JSON body as one normalized answer. */
export function parseOpenAIJson(body: Json): NormalizedAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : answerText(message.content);
  const toolCalls: { id: string; name: string; arguments: string }[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : 'call_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stopReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages JSON body as one normalized answer. */
export function parseAnthropicJson(body: Json): NormalizedAnswer | null {
  if (body.type !== 'message' || !Array.isArray(body.content)) return null;
  const toolCalls: { id: string; name: string; arguments: string }[] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      name: typeof block.name === 'string' ? block.name : '',
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(body.content),
    toolCalls,
    stopReason: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
  };
}
