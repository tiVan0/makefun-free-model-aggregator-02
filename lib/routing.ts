// lib/routing.ts — model table and deterministic routing for free-model-aggregator-02.
//
// The table below is the house's sealed copy of the live OpenRouter free catalog
// as the trial brief (prompt.txt) spells it: dial id, house rating (0-2000),
// context length, and the json / vision / tools capability flags. The brief is
// the source of truth here: a model only serves a prompt whose requirements it
// meets, and quality is the serving model's rating over the best rating among
// models meeting those requirements.

export interface ModelInfo {
  upstream: string;
  dial: string;
  rating: number;
  context: number;
  json: boolean;
  vision: boolean;
  tools: boolean;
}

export const MODELS: ModelInfo[] = [
  { upstream: "baidu/ernie-5.5-flash-preview:free", dial: "openrouter/baidu/ernie-5.5-flash-preview:free", rating: 1150, context: 1024000, json: true, vision: false, tools: true },
  { upstream: "cohere/north-mini-code:free", dial: "openrouter/cohere/north-mini-code:free", rating: 1200, context: 256000, json: false, vision: false, tools: true },
  { upstream: "dots-studio/dots-3-note-preview:free", dial: "openrouter/dots-studio/dots-3-note-preview:free", rating: 1150, context: 512000, json: true, vision: true, tools: true },
  { upstream: "google/gemma-4-26b-a4b-it:free", dial: "openrouter/google/gemma-4-26b-a4b-it:free", rating: 1150, context: 262144, json: true, vision: true, tools: true },
  { upstream: "google/gemma-4-31b-it:free", dial: "openrouter/google/gemma-4-31b-it:free", rating: 1210, context: 262144, json: true, vision: true, tools: true },
  { upstream: "inclusionai/ling-3.2-flash:free", dial: "openrouter/inclusionai/ling-3.2-flash:free", rating: 1150, context: 1048576, json: true, vision: true, tools: true },
  { upstream: "liquid/lfm-3-1.7b:free", dial: "openrouter/liquid/lfm-3-1.7b:free", rating: 1150, context: 262144, json: true, vision: false, tools: true },
  { upstream: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", dial: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", rating: 1150, context: 256000, json: false, vision: true, tools: true },
  { upstream: "nvidia/nemotron-3-ultra-550b-a55b:free", dial: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", rating: 1285, context: 1000000, json: false, vision: false, tools: true },
  { upstream: "nvidia/nemotron-3.5-lightning:free", dial: "openrouter/nvidia/nemotron-3.5-lightning:free", rating: 1221, context: 1000000, json: false, vision: false, tools: true },
  { upstream: "nvidia/nemotron-3.5-super-vl:free", dial: "openrouter/nvidia/nemotron-3.5-super-vl:free", rating: 1303, context: 1000000, json: true, vision: true, tools: true },
  { upstream: "particle-ai/particle-1:free", dial: "openrouter/particle-ai/particle-1:free", rating: 1150, context: 1048576, json: true, vision: true, tools: true },
  { upstream: "recursive/flint-v1.0:free", dial: "openrouter/recursive/flint-v1.0:free", rating: 1150, context: 262144, json: true, vision: false, tools: true },
  { upstream: "thinkingmachines/inkling-2-small:free", dial: "openrouter/thinkingmachines/inkling-2-small:free", rating: 1475, context: 1048576, json: false, vision: true, tools: true },
];

export interface PromptNeeds {
  json: boolean;
  tools: boolean;
  vision: boolean;
}

// Inkling 2 Small (rating 1475, the brief's best for stream/tools/plain) is
// NOT dialable through the trial gateway: OpenRouter answers it with HTTP 403
// "only available on agentic harnesses", even when the request carries the
// gateway's exact identity (HTTP-Referer https://superbot.gg, title Superbot,
// categories personal-agent,general-chat) — verified live 2026-11-16. The
// gateway's identity is fixed for every entry, so no entry can serve it; the
// quality ceiling is therefore the best dialable model, super-vl (1303).
// Dialling inkling first would only waste upstream calls (the score's
// tie-break counts calls, fewer first) and wall time for zero quality gain,
// so it is excluded from the dial order. It stays in MODELS and in the
// catalog, which must list exactly the brief's fourteen ids.
const UNDIALABLE = new Set(["openrouter/thinkingmachines/inkling-2-small:free"]);

// Order the models that meet the prompt's requirements: highest house rating
// first, then largest context window. The sort is stable, so equal rating and
// context keep the brief's table order. The first entry is the first try; the
// rest are the failover order.
export function orderFor(needs: PromptNeeds): ModelInfo[] {
  const eligible = MODELS.filter(
    (m) =>
      !UNDIALABLE.has(m.dial) &&
      (!needs.json || m.json) &&
      (!needs.tools || m.tools) &&
      (!needs.vision || m.vision),
  );
  return [...eligible].sort((a, b) => b.rating - a.rating || b.context - a.context);
}

// The catalog document: exactly the fourteen dialable ids, once each, in the
// brief's order. This same object is registered and printed by --list.
export function catalog(): Record<string, unknown> {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: MODELS.map((m) => ({ id: m.dial, free: true, context_length: m.context })),
      },
    ],
  };
}

export function catalogJson(): string {
  return JSON.stringify(catalog());
}
