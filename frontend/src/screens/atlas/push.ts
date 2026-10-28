/**
 * The push tab's request and its turns' labels (PushTab.tsx), kept apart from React so that
 * scripts/check_interface_defaults.py can bundle and run them: what POST /api/atlas/push is sent, and how each turn says
 * it was written. Greedy is the default and the reading; sampled, the three voices are drawn by the instrument's own
 * sampler at a temperature and a seed, the three sharing each draw (backend/sampler_request.py, atlas_ops.write_rows).
 */

/** How the next turn's voices are written: greedy, or sampled at `temperature` with `seed` (null: the server draws one
 * and the turn shows it). The temperature and the seed are kept while greedy, for the next sampled turn. */
export interface Writing {
  sampled: boolean;
  temperature: number;
  seed: number | null;
}

export const WRITING_DEFAULT: Writing = { sampled: false, temperature: 0.7, seed: null };

/** The server's bounds: a temperature above 0 samples (at most 2), and a seed is a whole number below 2^53. */
export const MAX_TEMPERATURE = 2;
export const MIN_TEMPERATURE = 0.05;
export const MAX_SEED = Number.MAX_SAFE_INTEGER;

/** How the server says a turn was written (the push's last event, `sampling`). */
export type Sampling =
  | "greedy"
  | {
      temperature: number;
      top_k: number;
      top_p: number;
      presence_penalty: number;
      seed: number;
      seed_drawn: boolean;
      samples: number;
      seeds: number[];
      draw: string;
    };

/** A fresh seed for the seed box ("new seed"), in the range the server draws its own from. */
export const newSeed = (): number => Math.floor(Math.random() * 2 ** 31);

/** The sampler sent: none when greedy (the server's default), else the temperature kept within the server's bounds
 * and the seed when there is one. No top-k, top-p or presence penalty: the tab samples from all the model's odds. */
export function samplerOf(w: Writing): { temperature: number; seed?: number } | undefined {
  if (!w.sampled) return undefined;
  const t = Number.isFinite(w.temperature) ? Math.min(MAX_TEMPERATURE, Math.max(MIN_TEMPERATURE, w.temperature)) : WRITING_DEFAULT.temperature;
  const out: { temperature: number; seed?: number } = { temperature: t };
  if (w.seed != null && Number.isFinite(w.seed)) out.seed = Math.min(MAX_SEED, Math.max(0, Math.round(w.seed)));
  return out;
}

/** The body of POST /api/atlas/push for one turn: the open neuron pushed by `dose` in `mode`, the earlier turns each
 * voice replays, and the sampler only when sampling. */
export function pushBody(
  focus: { layer: number; n: number },
  prompt: string,
  dose: number,
  mode: "add" | "mul",
  history: { user: string; replies: [string, string, string] }[],
  w: Writing,
): Record<string, unknown> {
  const body: Record<string, unknown> = { layer: focus.layer, n: focus.n, prompt, dose, mode, history };
  const sampler = samplerOf(w);
  if (sampler) body.sampler = sampler;
  return body;
}

/** How a turn was written, for its label: "greedy", or "T 0.7 · seed 123" ("drawn" when the server chose the seed).
 * From the server's word once the turn is done, else from what was asked. */
export function writtenAs(said: Sampling | null, asked: Writing): string {
  if (said === "greedy") return "greedy";
  if (said) return `T ${said.temperature} · seed ${said.seed}${said.seed_drawn ? " (drawn)" : ""}`;
  const s = samplerOf(asked);
  if (!s) return "greedy";
  return `T ${s.temperature} · ${s.seed != null ? `seed ${s.seed}` : "seed drawn by the server"}`;
}
