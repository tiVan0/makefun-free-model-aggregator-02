// lib/providers.ts — gateway client: Chat Completions calls plus SSE parsing.
//
// Every model is dialled on the gateway's Chat Completions route only:
// POST $OPENAI_BASE_URL/chat/completions with Authorization: Bearer
// $OPENAI_API_KEY (the trial's dummy key; the gateway swaps in the entrant's
// pushed OpenRouter key, which never enters the sandbox). A real provider
// 429/5xx arrives as sent, with its own body and no retry-after.

export interface GatewayFailure {
  status: number;
  code: string;
  model: string | null;
}

export interface GatewaySuccess {
  ok: true;
  status: number;
  json: unknown;
  text: string;
}

export interface GatewayRefusal {
  ok: false;
  status: number;
  code: string;
  body: string;
}

export type GatewayResult = GatewaySuccess | GatewayRefusal;

function errorCodeOf(text: string): string {
  try {
    const value = JSON.parse(text);
    if (value !== null && typeof value === "object") {
      const err = (value as Record<string, unknown>).error;
      if (err !== null && typeof err === "object") {
        const rec = err as Record<string, unknown>;
        if (typeof rec.code === "string" && rec.code !== "") return rec.code.slice(0, 64);
        if (typeof rec.type === "string" && rec.type !== "") return rec.type.slice(0, 64);
      }
    }
  } catch {
    // Not JSON: no code to read.
  }
  return "";
}

export async function callGateway(
  body: Record<string, unknown>,
  stream: boolean,
  timeoutMs = 25000,
): Promise<GatewayResult> {
  const base = (process.env.OPENAI_BASE_URL ?? "").replace(/\/$/, "");
  const key = process.env.OPENAI_API_KEY ?? "";
  const url = `${base}/chat/completions`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      return { ok: false, status: res.status, code: errorCodeOf(text), body: text };
    }
    let json: unknown = null;
    if (!stream) {
      try {
        json = JSON.parse(text);
      } catch {
        json = null;
      }
    }
    return { ok: true, status: res.status, json, text };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const timedOut = message.toLowerCase().includes("abort");
    return { ok: false, status: 0, code: timedOut ? "timeout" : "network_error", body: "" };
  } finally {
    clearTimeout(timer);
  }
}

// Parse a server-sent-events body into its JSON payloads, in order. The
// trailing "[DONE]" frame carries no payload and is skipped.
export function parseSse(text: string): Array<Record<string, unknown>> {
  const chunks: Array<Record<string, unknown>> = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const frame of normalized.split("\n\n")) {
    if (frame.trim() === "") continue;
    for (const line of frame.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice("data:".length).replace(/^ /, "");
      if (payload === "[DONE]" || payload === "") continue;
      try {
        const value = JSON.parse(payload);
        if (value !== null && typeof value === "object" && !Array.isArray(value)) {
          chunks.push(value as Record<string, unknown>);
        }
      } catch {
        // Skip a malformed frame; the stream's other frames still stand.
      }
    }
  }
  return chunks;
}
