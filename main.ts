// main.ts — free-model-aggregator-02 entrypoint.
//
// One Node 22 stdlib-only server: it serves OpenAI Chat Completions and
// Anthropic Messages behind one aggregator, dials OpenRouter's fourteen live
// :free models through the grader's gateway, registers with the env, and
// exits 0 when the env's work is done. Every decision is a function of the
// requests seen and the gateway answers seen so far — never of the wall
// clock. The only sleeps are fixed retry and poll delays, which change no
// decision and reach no export.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { MODELS, catalog, catalogJson, orderFor } from "./lib/routing.ts";
import type { ModelInfo, PromptNeeds } from "./lib/routing.ts";
import { callGateway, parseSse } from "./lib/providers.ts";
import type { GatewayFailure } from "./lib/providers.ts";
import {
  anthropicToOpenAI,
  estimateTokens,
  needsOf,
  openAIChunksToAnthropicSse,
  openAIChunksToOpenAISse,
  openAIRequestBody,
  openAIToAnthropic,
  syntheticChunksFromJson,
} from "./lib/wire.ts";

const FAILURES_HEADER = "x-makefun-upstream-failures";
const RETRY_DELAY_MS = 800;
const POLL_MS = 1000;
const MAX_BODY_BYTES = 8 * 1024 * 1024;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

// Read a request body whole, capped so a runaway body cannot grow the heap.
async function readBody(req: IncomingMessage): Promise<string | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const part = chunk as Buffer;
    size += part.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(part);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function failuresHeaderValue(failures: GatewayFailure[]): string | null {
  if (failures.length === 0) return null;
  const trail = failures.slice(0, 8).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  return JSON.stringify(trail);
}

function sendJson(res: ServerResponse, status: number, value: unknown, failures: GatewayFailure[]): void {
  const headers: Record<string, string> = { "content-type": "application/json" };
  const trail = failuresHeaderValue(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  const body = Buffer.from(`${JSON.stringify(value)}\n`);
  res.writeHead(status, { ...headers, "content-length": String(body.length) });
  res.end(body);
}

function sendSse(res: ServerResponse, frames: string, failures: GatewayFailure[]): void {
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
  const trail = failuresHeaderValue(failures);
  if (trail !== null) headers[FAILURES_HEADER] = trail;
  res.writeHead(200, headers);
  res.end(frames);
}

interface Served {
  ok: boolean;
  dialId: string | null;
  json: unknown;
  sseText: string | null;
  failures: GatewayFailure[];
}

// Dial the candidates in order with failover: on a gateway 429/5xx, retry
// the same model once after a fixed delay, then move to the next model. At
// most three models are tried per prompt, and no model is retried more than
// once. A refusal that is not a 429/5xx is recorded and the next model is
// tried without a retry, since the same request would fail the same way.
// Every refusal is recorded in order for the failures header.
async function dialWithFailover(
  candidates: ModelInfo[],
  buildBody: (model: string) => Record<string, unknown>,
  stream: boolean,
): Promise<Served> {
  const failures: GatewayFailure[] = [];
  const tried = candidates.slice(0, 3);
  for (const candidate of tried) {
    for (let attempt = 0; attempt < 2; attempt++) {
      if (attempt === 1) await sleep(RETRY_DELAY_MS);
      const result = await callGateway(buildBody(candidate.dial), stream);
      if (result.ok) {
        return { ok: true, dialId: candidate.dial, json: result.json, sseText: stream ? result.text : null, failures };
      }
      failures.push({ status: result.status, code: result.code, model: candidate.dial });
      const retryable = result.status === 429 || (result.status >= 500 && result.status < 600);
      if (!retryable) break;
    }
  }
  return { ok: false, dialId: null, json: null, sseText: null, failures };
}

// A non-stream Chat Completions answer with the model field rewritten to
// the id that was really dialled and usage passed through whole (synthesised
// only when the provider sent none).
function openAIAnswer(resp: unknown, dialId: string): Record<string, unknown> {
  const root = isRecord(resp) ? resp : {};
  const choices = Array.isArray(root.choices) ? root.choices : [];
  const usage = isRecord(root.usage) ? { ...root.usage } : {};
  let outputTokens = typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0;
  if (outputTokens < 1) {
    const first = choices.length > 0 && isRecord(choices[0]) ? (choices[0] as Record<string, unknown>) : {};
    const message = isRecord(first.message) ? first.message : {};
    const content = typeof message.content === "string" ? message.content : "";
    outputTokens = estimateTokens(content);
    usage.completion_tokens = outputTokens;
  }
  if (typeof usage.prompt_tokens !== "number") usage.prompt_tokens = 0;
  return {
    id: typeof root.id === "string" ? root.id : "chatcmpl_1",
    object: "chat.completion",
    created: typeof root.created === "number" ? root.created : 0,
    model: dialId,
    choices,
    usage,
  };
}

// The model list: the OpenAI shape for an Authorization: Bearer request,
// the Anthropic shape for an x-api-key request. Both list exactly the
// fourteen dialable ids, once each.
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const auth = req.headers.authorization ?? "";
  const apiKey = req.headers["x-api-key"];
  if (typeof apiKey === "string" && apiKey !== "" && !auth.startsWith("Bearer ")) {
    sendJson(res, 200, {
      data: MODELS.map((m) => ({ id: m.dial, type: "model", free: true })),
      has_more: false,
      first_id: MODELS[0].dial,
      last_id: MODELS[MODELS.length - 1].dial,
    }, []);
    return;
  }
  sendJson(res, 200, {
    object: "list",
    data: MODELS.map((m) => ({ id: m.dial, object: "model", created: 0, owned_by: "openrouter", free: true })),
  }, []);
}

function upstreamErrorBody(): Record<string, unknown> {
  return { error: { message: "the upstream model could not serve this prompt", type: "server_error", code: "upstream_failed" } };
}

async function handleChatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const raw = await readBody(req);
  if (raw === null) {
    sendJson(res, 413, { error: { message: "request body too large", type: "invalid_request_error" } }, []);
    return;
  }
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    sendJson(res, 400, { error: { message: "request body is not JSON", type: "invalid_request_error" } }, []);
    return;
  }
  if (!isRecord(body)) {
    sendJson(res, 400, { error: { message: "request body is not one JSON object", type: "invalid_request_error" } }, []);
    return;
  }
  const needs: PromptNeeds = needsOf(body, "openai");
  const candidates = orderFor(needs);
  const stream = body.stream === true;
  const served = await dialWithFailover(
    candidates,
    (model) => openAIRequestBody(body, model),
    stream,
  );
  if (!served.ok || served.dialId === null) {
    sendJson(res, 502, upstreamErrorBody(), served.failures);
    return;
  }
  if (stream) {
    let chunks = parseSse(served.sseText ?? "");
    if (chunks.length === 0 && served.json !== null) {
      chunks = syntheticChunksFromJson(served.json);
    } else if (chunks.length === 0 && served.sseText !== null && served.sseText !== "") {
      try {
        chunks = syntheticChunksFromJson(JSON.parse(served.sseText));
      } catch {
        chunks = [];
      }
    }
    if (chunks.length === 0) {
      sendJson(res, 502, upstreamErrorBody(), served.failures);
      return;
    }
    sendSse(res, openAIChunksToOpenAISse(chunks, served.dialId), served.failures);
    return;
  }
  if (served.json === null) {
    sendJson(res, 502, upstreamErrorBody(), served.failures);
    return;
  }
  sendJson(res, 200, openAIAnswer(served.json, served.dialId), served.failures);
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const raw = await readBody(req);
  if (raw === null) {
    sendJson(res, 413, { error: { message: "request body too large", type: "invalid_request_error" } }, []);
    return;
  }
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    sendJson(res, 400, { error: { message: "request body is not JSON", type: "invalid_request_error" } }, []);
    return;
  }
  if (!isRecord(body)) {
    sendJson(res, 400, { error: { message: "request body is not one JSON object", type: "invalid_request_error" } }, []);
    return;
  }
  const needs: PromptNeeds = needsOf(body, "anthropic");
  const candidates = orderFor(needs);
  const stream = body.stream === true;
  const served = await dialWithFailover(
    candidates,
    (model) => anthropicToOpenAI(body, model),
    stream,
  );
  if (!served.ok || served.dialId === null) {
    sendJson(res, 502, upstreamErrorBody(), served.failures);
    return;
  }
  if (stream) {
    let chunks = parseSse(served.sseText ?? "");
    if (chunks.length === 0 && served.json !== null) {
      chunks = syntheticChunksFromJson(served.json);
    } else if (chunks.length === 0 && served.sseText !== null && served.sseText !== "") {
      try {
        chunks = syntheticChunksFromJson(JSON.parse(served.sseText));
      } catch {
        chunks = [];
      }
    }
    if (chunks.length === 0) {
      sendJson(res, 502, upstreamErrorBody(), served.failures);
      return;
    }
    sendSse(res, openAIChunksToAnthropicSse(chunks, served.dialId), served.failures);
    return;
  }
  if (served.json === null) {
    sendJson(res, 502, upstreamErrorBody(), served.failures);
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.json, served.dialId), served.failures);
}

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const url = new URL(req.url ?? "/", "http://server");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      handleModels(req, res);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      await handleChatCompletions(req, res);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      await handleMessages(req, res);
      return;
    }
    sendJson(res, 404, { error: { message: `no route ${req.method ?? ""} ${url.pathname}`, type: "invalid_request_error" } }, []);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    try {
      sendJson(res, 500, { error: { message, type: "server_error" } }, []);
    } catch {
      // The response is already gone; nothing left to do.
    }
  }
}

async function register(envUrl: string, port: number): Promise<boolean> {
  const catalogDoc = catalog();
  for (let attempt = 0; attempt < 12; attempt++) {
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog: catalogDoc }),
      });
      if (res.ok) return true;
      if (res.status >= 400 && res.status < 500) return false;
    } catch {
      // The env is not up yet; wait and try again.
    }
    await sleep(500);
  }
  return false;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes("--list")) {
    process.stdout.write(`${catalogJson()}\n`);
    return;
  }
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  if (envUrl === "") {
    // Outside a trial there is no env to register with: print the catalog
    // and exit 0, as the starter does.
    process.stdout.write(`${catalogJson()}\n`);
    return;
  }
  const server = createServer((req, res) => {
    void handleRequest(req, res);
  });
  process.on("SIGTERM", () => {
    try {
      server.close();
    } catch {
      // Already closed.
    }
    process.exit(0);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (port === 0) {
    process.stderr.write("the server did not bind a port\n");
    process.exit(1);
  }
  const registered = await register(envUrl, port);
  if (!registered) {
    process.stderr.write("failed to register with the env\n");
    process.exit(1);
  }
  for (;;) {
    await sleep(POLL_MS);
    try {
      const res = await fetch(`${envUrl.replace(/\/$/, "")}/work`);
      const value = (await res.json()) as unknown;
      if (isRecord(value) && value.done === true) break;
    } catch {
      // Keep polling; a transient env error is not the trial ending.
    }
  }
  server.close();
  process.exit(0);
}

main().catch((err) => {
  process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
