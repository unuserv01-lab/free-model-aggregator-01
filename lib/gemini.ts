// The Gemini wire: generateContent and streamGenerateContent.
//
// A Gemini-wire provider answers POST <base>/v1beta/models/<model>:generateContent (or
// :streamGenerateContent with ?alt=sse for a streamed answer) and GET <base>/v1beta/models. The request
// body is { systemInstruction, contents, tools, generationConfig }; the answer is { candidates: [
// { content: { parts }, finishReason } ], usageMetadata }.
//
// Translation in, from either client wire:
//   - the system prompt becomes systemInstruction.parts;
//   - messages become contents: user parts are text and inlineData images (a data: URL) or fileData
//     (an http(s) URL); assistant parts are text and functionCall for tool_calls; tool results become
//     functionResponse parts, attached to the most recent user content, so the provider's echo of the
//     user text still reads the client's user text and not an empty tool-only content;
//   - tools become one tools entry of functionDeclarations; the output cap becomes
//     generationConfig.maxOutputTokens, stop sequences become stopSequences, and a JSON-mode request
//     becomes responseMimeType application/json.
// Translation out: the first candidate's parts give the text, its finishReason the stop, and
// usageMetadata the token counts; a stream's frames are read whole and normalized the same way.

import { isRecord, type Json } from './providers.ts';
import type { NormalizedAnswer } from './wire.ts';

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One OpenAI content part as a Gemini part: text or an inline image; null for the rest. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part: text, an inline image, or a tool result; null for the rest. */
function geminiPartOfAnthropic(block: Json): { part: Json; toolResult: boolean } | null {
  if (block.type === 'text' && typeof block.text === 'string') return { part: { text: block.text }, toolResult: false };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { part: { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } }, toolResult: false };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { part: { fileData: { fileUri: source.url } }, toolResult: false };
  }
  if (block.type === 'tool_result') {
    const content = typeof block.content === 'string' ? block.content : textOf(block.content);
    return {
      part: { functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'tool', response: { content } } },
      toolResult: true,
    };
  }
  return null;
}

interface GeminiBody {
  systemInstruction: Json | null;
  contents: Json[];
  tools: Json[];
  generationConfig: Json;
}

/** The shared tail of a Gemini request body: tools, the output cap, stops, temperature and JSON mode. */
function geminiTail(body: Json, jsonMode: boolean): { tools: Json[]; generationConfig: Json } {
  const tools: Json[] = [];
  const declarations: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      const fn = isRecord(raw) && isRecord(raw.function) ? raw.function : isRecord(raw) && typeof raw.name === 'string' ? raw : null;
      if (fn === null) continue;
      const name = typeof fn.name === 'string' ? fn.name : '';
      if (name === '') continue;
      declarations.push({
        name,
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : isRecord(fn.input_schema) ? fn.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  if (declarations.length > 0) tools.push({ functionDeclarations: declarations });
  const generationConfig: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) generationConfig.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' && body.stop !== '' ? [body.stop] : [];
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (jsonMode) generationConfig.responseMimeType = 'application/json';
  return { tools, generationConfig };
}

/** The contents list, with consecutive same-role contents merged and tool results attached to the last user content. */
function mergeContents(contents: Json[]): Json[] {
  const out: Json[] = [];
  for (const content of contents) {
    const last = out[out.length - 1];
    if (last !== undefined && last.role === content.role && Array.isArray(last.parts) && Array.isArray(content.parts)) {
      last.parts.push(...(content.parts as Json[]));
    } else {
      out.push({ role: content.role, parts: [...(Array.isArray(content.parts) ? content.parts : [])] });
    }
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. `jsonMode` is true when the
 * client asked for JSON mode (response_format json_object or json_schema).
 */
export function openAIToGemini(body: Json, jsonMode: boolean): Json {
  const built: GeminiBody = { systemInstruction: null, contents: [], tools: [], generationConfig: {} };
  const systemParts: string[] = [];
  const rawContents: Json[] = [];
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
      const parts: Json[] = [];
      const text = typeof content === 'string' ? content : textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty call.
          }
        } else if (isRecord(call.function.arguments)) args = call.function.arguments;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) rawContents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      // A tool result belongs to the most recent user content, so the provider's echo of the user text
      // still reads the client's user text.
      const text = typeof content === 'string' ? content : textOf(content);
      const part: Json = { functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : 'tool', response: { content: text } } };
      let target: Json | undefined;
      for (let i = rawContents.length - 1; i >= 0; i--) {
        if (rawContents[i]?.role === 'user') {
          target = rawContents[i];
          break;
        }
      }
      if (target === undefined) {
        target = { role: 'user', parts: [] };
        rawContents.push(target);
      }
      (target.parts as Json[]).push(part);
      continue;
    }
    // A user message: string content, or parts of text and images.
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        const converted = geminiPartOfOpenAI(part);
        if (converted !== null) parts.push(converted);
      }
    }
    if (parts.length > 0) rawContents.push({ role: 'user', parts });
  }
  if (systemParts.length > 0) built.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  built.contents = mergeContents(rawContents);
  const tail = geminiTail(body, jsonMode);
  built.tools = tail.tools;
  built.generationConfig = tail.generationConfig;
  const out: Json = { contents: built.contents };
  if (built.systemInstruction !== null) out.systemInstruction = built.systemInstruction;
  if (built.tools.length > 0) out.tools = built.tools;
  if (Object.keys(built.generationConfig).length > 0) out.generationConfig = built.generationConfig;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body (no JSON mode on this wire). */
export function anthropicToGemini(body: Json): Json {
  const built: GeminiBody = { systemInstruction: null, contents: [], tools: [], generationConfig: {} };
  const system = textOf(body.system);
  if (system !== '') built.systemInstruction = { parts: [{ text: system }] };
  const rawContents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') rawContents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    const toolResults: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (assistant && block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      const converted = geminiPartOfAnthropic(block);
      if (converted === null) continue;
      if (converted.toolResult) toolResults.push(converted.part);
      else parts.push(converted.part);
    }
    if (parts.length > 0) rawContents.push({ role: assistant ? 'model' : 'user', parts });
    for (const part of toolResults) {
      let target: Json | undefined;
      for (let i = rawContents.length - 1; i >= 0; i--) {
        if (rawContents[i]?.role === 'user') {
          target = rawContents[i];
          break;
        }
      }
      if (target === undefined) {
        target = { role: 'user', parts: [] };
        rawContents.push(target);
      }
      (target.parts as Json[]).push(part);
    }
  }
  built.contents = mergeContents(rawContents);
  const tail = geminiTail(body, false);
  built.tools = tail.tools;
  built.generationConfig = tail.generationConfig;
  const out: Json = { contents: built.contents };
  if (built.systemInstruction !== null) out.systemInstruction = built.systemInstruction;
  if (built.tools.length > 0) out.tools = built.tools;
  if (Object.keys(built.generationConfig).length > 0) out.generationConfig = built.generationConfig;
  return out;
}

/** The text of a candidate's parts: its text parts joined, '' when there are none. */
function candidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** The tool calls of a candidate's parts: its functionCall parts, in order. */
function candidateToolCalls(candidate: Json): { id: string; name: string; arguments: string }[] {
  const content = isRecord(candidate.content) ? candidate.content : null;
  const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
  const calls: { id: string; name: string; arguments: string }[] = [];
  parts.forEach((part, index) => {
    if (!isRecord(part) || !isRecord(part.functionCall)) return;
    const fn = part.functionCall;
    calls.push({
      id: `call_${index}`,
      name: typeof fn.name === 'string' ? fn.name : '',
      arguments: JSON.stringify(isRecord(fn.args) ? fn.args : {}),
    });
  });
  return calls;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One Gemini generateContent JSON body as a normalized answer; null when it holds no candidate. */
export function parseGeminiJson(body: Json): NormalizedAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: candidateText(candidate),
    toolCalls: candidateToolCalls(candidate),
    stopReason: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    usage: { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** One Gemini streamGenerateContent SSE text as a normalized answer. */
export function parseGeminiStream(text: string): NormalizedAnswer | null {
  const normalized = text.replace(/\r\n/g, '\n');
  let outText = '';
  let stopReason: string | null = null;
  let input = 0;
  let output = 0;
  const toolCalls: { id: string; name: string; arguments: string }[] = [];
  let sawCandidate = false;
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (dataLines.length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(dataLines.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      sawCandidate = true;
      outText += candidateText(candidate);
      toolCalls.push(...candidateToolCalls(candidate));
      if (typeof candidate.finishReason === 'string') stopReason = candidate.finishReason;
    }
    if (isRecord(value.usageMetadata)) {
      input = numberOrZero(value.usageMetadata.promptTokenCount);
      output = numberOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawCandidate) return null;
  return { text: outText, toolCalls, stopReason, usage: { input, output } };
}
