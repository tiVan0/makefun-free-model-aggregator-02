// lib/wire.ts — OpenAI <-> Anthropic translation, both directions, plus SSE
// framing. Nothing here touches answer content: the prompt's unique token
// travels verbatim from the provider's answer to the client's wire.

export function maxTokensOf(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.min(8192, Math.max(1, Math.floor(value)));
  }
  return 1024;
}

export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function anthropicTextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
        parts.push(block.text);
      }
    }
    return parts.join("");
  }
  return "";
}

function openAITextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const part of content) {
      if (isRecord(part) && part.type === "text" && typeof part.text === "string") {
        parts.push(part.text);
      }
    }
    return parts.join("");
  }
  return "";
}

// What a client body requires of its serving model: JSON mode, tools, vision.
export function needsOf(body: Record<string, unknown>, api: "openai" | "anthropic"): {
  json: boolean;
  tools: boolean;
  vision: boolean;
} {
  let json = false;
  if (api === "openai" && isRecord(body.response_format)) {
    json = body.response_format.type === "json_object" || body.response_format.type === "json_schema";
  }
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === "image_url" || part.type === "image") vision = true;
    }
  }
  return { json, tools, vision };
}

// Build the Chat Completions body for an OpenAI-shaped client request: the
// client's messages, tools, response_format, max_tokens and temperature pass
// through unchanged; only the model is replaced by the caller.
export function openAIRequestBody(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const out: Record<string, unknown> = {
    model,
    messages: Array.isArray(body.messages) ? body.messages : [],
  };
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (isRecord(body.response_format)) out.response_format = body.response_format;
  out.max_tokens = maxTokensOf(body.max_tokens);
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// Translate an Anthropic Messages request into a Chat Completions body:
// system becomes a system message, messages and tools carry over in the
// OpenAI shape, max_tokens and temperature pass through.
export function anthropicToOpenAI(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  const system = body.system;
  if (typeof system === "string" && system !== "") {
    messages.push({ role: "system", content: system });
  } else if (Array.isArray(system)) {
    const text = anthropicTextOf(system);
    if (text !== "") messages.push({ role: "system", content: text });
  }
  const input = Array.isArray(body.messages) ? body.messages : [];
  for (const message of input) {
    if (!isRecord(message)) continue;
    const role = message.role;
    if (role !== "user" && role !== "assistant") continue;
    messages.push({ role, content: anthropicTextOf(message.content) });
  }
  const out: Record<string, unknown> = { model, messages };
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = (body.tools as Array<Record<string, unknown>>).map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : "",
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: "object", properties: {} },
      },
    }));
  }
  out.max_tokens = maxTokensOf(body.max_tokens);
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

const OPENAI_TO_ANTHROPIC_STOP: Record<string, string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  function_call: "tool_use",
  content_filter: "refusal",
};

export function mapStopReason(finishReason: unknown): string {
  if (typeof finishReason === "string" && OPENAI_TO_ANTHROPIC_STOP[finishReason] !== undefined) {
    return OPENAI_TO_ANTHROPIC_STOP[finishReason];
  }
  return "end_turn";
}

// Translate a Chat Completions answer back into a Messages answer: content
// blocks (text first, then any tool_use blocks), the mapped stop reason, and
// usage as input_tokens/output_tokens. The model field is always the id that
// was really dialled.
export function openAIToAnthropic(resp: unknown, dialId: string): Record<string, unknown> {
  const root = isRecord(resp) ? resp : {};
  const choices = Array.isArray(root.choices) ? root.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? (choices[0] as Record<string, unknown>) : {};
  const message = isRecord(first.message) ? first.message : {};
  const content: Array<Record<string, unknown>> = [];
  const text = openAITextOf(message.content);
  if (text !== "") content.push({ type: "text", text });
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call)) continue;
      const fn = isRecord(call.function) ? call.function : {};
      let input: unknown = {};
      if (typeof fn.arguments === "string") {
        try {
          input = JSON.parse(fn.arguments);
        } catch {
          input = {};
        }
      }
      content.push({
        type: "tool_use",
        id: typeof call.id === "string" ? call.id : "toolu_1",
        name: typeof fn.name === "string" ? fn.name : "",
        input,
      });
    }
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const usage = isRecord(root.usage) ? root.usage : {};
  const inputTokens = typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0;
  let outputTokens = typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0;
  if (outputTokens < 1) {
    const plain = content
      .filter((block) => block.type === "text")
      .map((block) => String(block.text ?? ""))
      .join("");
    outputTokens = estimateTokens(plain);
  }
  return {
    id: typeof root.id === "string" ? root.id : "msg_1",
    type: "message",
    role: "assistant",
    model: dialId,
    content,
    stop_reason: mapStopReason(first.finish_reason),
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

// Re-emit buffered OpenAI SSE chunks as an OpenAI event stream, with the
// model field rewritten to the dialled id on every chunk, then [DONE].
export function openAIChunksToOpenAISse(chunks: Array<Record<string, unknown>>, dialId: string): string {
  const frames: string[] = [];
  for (const chunk of chunks) {
    const rewritten = { ...chunk, model: dialId };
    frames.push(`data: ${JSON.stringify(rewritten)}\n\n`);
  }
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

// Convert buffered OpenAI SSE chunks into an Anthropic event stream: one
// message_start (carrying the input usage), one text content block with a
// delta per text fragment, then message_delta (stop reason + output usage)
// and message_stop, in that order.
export function openAIChunksToAnthropicSse(
  chunks: Array<Record<string, unknown>>,
  dialId: string,
): string {
  let messageId = "msg_1";
  let finishReason: unknown = null;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  const deltas: string[] = [];
  for (const chunk of chunks) {
    if (typeof chunk.id === "string" && chunk.id !== "") messageId = chunk.id;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? (choices[0] as Record<string, unknown>) : null;
    if (first !== null) {
      const delta = isRecord(first.delta) ? first.delta : {};
      if (typeof delta.content === "string" && delta.content !== "") deltas.push(delta.content);
      if (typeof first.finish_reason === "string" && first.finish_reason !== "") {
        finishReason = first.finish_reason;
      }
    }
    if (isRecord(chunk.usage)) {
      if (typeof chunk.usage.prompt_tokens === "number") inputTokens = chunk.usage.prompt_tokens;
      if (typeof chunk.usage.completion_tokens === "number") outputTokens = chunk.usage.completion_tokens;
    }
  }
  const text = deltas.join("");
  if (inputTokens === null) inputTokens = 0;
  if (outputTokens === null || outputTokens < 1) outputTokens = estimateTokens(text);
  const events: string[] = [];
  events.push(
    `event: message_start\ndata: ${JSON.stringify({
      type: "message_start",
      message: {
        id: messageId,
        type: "message",
        role: "assistant",
        model: dialId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: inputTokens, output_tokens: 0 },
      },
    })}\n\n`,
  );
  events.push(
    `event: content_block_start\ndata: ${JSON.stringify({
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    })}\n\n`,
  );
  for (const fragment of deltas) {
    events.push(
      `event: content_block_delta\ndata: ${JSON.stringify({
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: fragment },
      })}\n\n`,
    );
  }
  events.push(
    `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`,
  );
  events.push(
    `event: message_delta\ndata: ${JSON.stringify({
      type: "message_delta",
      delta: { stop_reason: mapStopReason(finishReason) },
      usage: { output_tokens: outputTokens },
    })}\n\n`,
  );
  events.push(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
  return events.join("");
}

// When a stream was asked for but the gateway answered one plain JSON
// completion instead, shape it as a one-chunk stream so the same emitters
// can serve it.
export function syntheticChunksFromJson(resp: unknown): Array<Record<string, unknown>> {
  const root = isRecord(resp) ? resp : {};
  const choices = Array.isArray(root.choices) ? root.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? (choices[0] as Record<string, unknown>) : {};
  const message = isRecord(first.message) ? first.message : {};
  const text = openAITextOf(message.content);
  const id = typeof root.id === "string" ? root.id : "chatcmpl_1";
  const created = typeof root.created === "number" ? root.created : 0;
  const usage = isRecord(root.usage) ? root.usage : {};
  const chunks: Array<Record<string, unknown>> = [];
  if (text !== "") {
    chunks.push({
      id,
      object: "chat.completion.chunk",
      created,
      model: "",
      choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: null }],
    });
  }
  chunks.push({
    id,
    object: "chat.completion.chunk",
    created,
    model: "",
    choices: [{ index: 0, delta: {}, finish_reason: typeof first.finish_reason === "string" ? first.finish_reason : "stop" }],
  });
  chunks.push({
    id,
    object: "chat.completion.chunk",
    created,
    model: "",
    choices: [],
    usage: {
      prompt_tokens: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
      completion_tokens: typeof usage.completion_tokens === "number" ? usage.completion_tokens : estimateTokens(text),
    },
  });
  return chunks;
}
