import type { EmbryoMlAssessment } from '../types';

const API_NOTE =
  'Research / demo only. Predicts live-birth class, not Gardner grade or developmental stage.';

export type PredictLiveBirthResponse = {
  predicted_lb_class: 0 | 1;
  probability_lb_1: number;
  probabilities: [number, number];
  model_type?: string;
  note?: string;
};

function trimBase(url: string): string {
  return url.replace(/\/$/, '');
}

/** Convert a data URL or blob URL to a Blob for multipart upload. */
export async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl);
  return res.blob();
}

export async function predictLiveBirth(
  baseUrl: string,
  imageBlob: Blob,
  opts?: { age?: number; exp?: number; icm?: number; te?: number; d?: number }
): Promise<PredictLiveBirthResponse> {
  const base = trimBase(baseUrl);
  const form = new FormData();
  form.append('file', imageBlob, 'embryo.jpg');
  if (opts?.exp != null) form.append('exp', String(opts.exp));
  if (opts?.icm != null) form.append('icm', String(opts.icm));
  if (opts?.te != null) form.append('te', String(opts.te));
  if (opts?.d != null) form.append('d', String(opts.d));
  if (opts?.age != null) form.append('age', String(opts.age));

  const res = await fetch(`${base}/predict`, { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  const data = (await res.json()) as PredictLiveBirthResponse;
  return { ...data, note: data.note ?? API_NOTE };
}

export function mlApiBase(): string | undefined {
  const u = import.meta.env.VITE_ML_API_URL;
  return typeof u === 'string' && u.trim() ? trimBase(u.trim()) : undefined;
}

/**
 * Call predict API for a File (e.g. first upload). Returns undefined if VITE_ML_API_URL is not set.
 */
export async function fetchMlAssessmentForImage(
  file: File,
  opts?: { age?: number }
): Promise<EmbryoMlAssessment | undefined> {
  const base = mlApiBase();
  if (!base) return undefined;
  const assessedAt = new Date().toISOString();
  try {
    const data = await predictLiveBirth(base, file, { age: opts?.age });
    return {
      assessedAt,
      predictedLbClass: data.predicted_lb_class,
      probabilityLb1: data.probability_lb_1,
      probabilities: data.probabilities,
    };
  } catch (e) {
    return {
      assessedAt,
      error: e instanceof Error ? e.message : 'Prediction failed',
    };
  }
}
