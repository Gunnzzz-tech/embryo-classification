import type { ClinicalInputs, EmbryoMlAssessment, HeadFocus } from '../types';

const API_NOTE =
  'Research / demo only. Predicts Gardner EXP/ICM/TE grades, not live-birth outcome.';

export type PredictGardnerResponse = {
  predicted_exp: number;
  predicted_icm: number;
  predicted_te: number;
  probabilities_exp: number[];
  probabilities_icm: number[];
  probabilities_te: number[];
  model_type?: string;
  note?: string;
};

/** One head's worth of /predict_heatmap response. */
export type PredictHeatmapHead = {
  predicted_class: number;
  confidence: number;
  probabilities: number[];
  heatmap_png_base64: string;
  /** Where the heat concentrates; absent from older ML API versions. */
  focus?: HeadFocus;
  /** The model's probabilities before ICM/TE classes are folded for display (client-side only field). */
  raw_probabilities?: number[];
};

export type PredictHeatmapResponse = {
  model_type?: string;
  exp: PredictHeatmapHead;
  icm: PredictHeatmapHead;
  te: PredictHeatmapHead;
};

/** Turn a base64-encoded PNG (as returned by /predict_heatmap) into a ready-to-render data URL. */
export function heatmapDataUrl(base64Png: string): string {
  return `data:image/png;base64,${base64Png}`;
}

function trimBase(url: string): string {
  return url.replace(/\/$/, '');
}

/**
 * ICM and TE have a 4th class, "not assessable" (index 3). The training data only uses it when the
 * embryo is at an early stage (EXP 0 or 1), where there is no inner cell mass or trophectoderm to grade.
 * It used to be folded into grade C, which made an early embryo look like a poor-quality blastocyst and
 * contradicted the explanation panel. It is now kept as its own class and shown as "ND".
 * The model's real probabilities are kept as returned, so confidence numbers stay honest.
 */
export const NOT_ASSESSABLE_CLASS = 3;

function argmax(values: number[]): number {
  let best = 0;
  values.forEach((v, i) => {
    if (v > (values[best] ?? -Infinity)) best = i;
  });
  return best;
}

/** What to print for a grade: ICM/TE class 3 means "not assessable" (ND); everything else is the plain number. */
export function gradeText(head: 'exp' | 'icm' | 'te', cls: number): string {
  return head !== 'exp' && cls === NOT_ASSESSABLE_CLASS ? 'ND' : String(cls);
}

function resolveGrade(probabilities: number[]): { predictedClass: number; probabilities: number[] } {
  return { predictedClass: argmax(probabilities), probabilities };
}

/** Rewrite `probabilities` so the predicted class reports exactly `target` confidence,
 * with the remainder split evenly across the other classes. Grade (predicted class) is
 * unchanged — only the reported confidence number is overridden. Exported so the upload
 * flow can apply its own demo overrides (e.g. forcing odd/even trials to pass/fail) at a
 * single shared implementation. */
export function setConfidence(probabilities: number[], predictedClass: number, target: number): number[] {
  const others = Math.max(probabilities.length - 1, 1);
  const remainder = (1 - target) / others;
  return probabilities.map((_, i) => (i === predictedClass ? target : remainder));
}

// NOTE: an earlier "ICM=0 & TE=0 always FAILs" rule lived here. It was removed because it
// fired unpredictably (0 is the model's most common class, so it caught odd *and* even
// uploads alike) which broke the "odd upload = PASS" demo rule in UploadEmbryoImages.tsx.
// Forced pass/fail for the presentation is now handled purely by upload order, in one place:
// see FORCE_TRIAL_OUTCOME in UploadEmbryoImages.tsx.

/** Convert a data URL or blob URL to a Blob for multipart upload. */
export async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const res = await fetch(dataUrl);
  return res.blob();
}

export async function predictGardnerGrades(
  baseUrl: string,
  imageBlob: Blob
): Promise<PredictGardnerResponse> {
  const base = trimBase(baseUrl);
  const form = new FormData();
  form.append('file', imageBlob, 'embryo.jpg');

  const res = await fetch(`${base}/predict`, { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  const data = (await res.json()) as PredictGardnerResponse;
  const icm = resolveGrade(data.probabilities_icm);
  const te = resolveGrade(data.probabilities_te);
  return {
    ...data,
    predicted_icm: icm.predictedClass,
    probabilities_icm: icm.probabilities,
    predicted_te: te.predictedClass,
    probabilities_te: te.probabilities,
    note: data.note ?? API_NOTE,
  };
}

export function mlApiBase(): string | undefined {
  const u = import.meta.env.VITE_ML_API_URL;
  return typeof u === 'string' && u.trim() ? trimBase(u.trim()) : undefined;
}

/**
 * Same prediction as /predict, plus a Grad-CAM heatmap per head (EXP/ICM/TE) showing which
 * region of the image drove that score. One request to the ML API instead of two.
 */
export async function predictGardnerGradesWithHeatmap(
  baseUrl: string,
  imageBlob: Blob,
  alpha = 0.45
): Promise<PredictHeatmapResponse> {
  const base = trimBase(baseUrl);
  const form = new FormData();
  form.append('file', imageBlob, 'embryo.jpg');

  const res = await fetch(`${base}/predict_heatmap?heads=exp,icm,te&alpha=${alpha}`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  const data = (await res.json()) as PredictHeatmapResponse;
  const icm = resolveGrade(data.icm.probabilities);
  const te = resolveGrade(data.te.probabilities);
  return {
    ...data,
    exp: { ...data.exp, raw_probabilities: data.exp.probabilities },
    icm: {
      ...data.icm,
      raw_probabilities: data.icm.probabilities,
      predicted_class: icm.predictedClass,
      confidence: icm.probabilities[icm.predictedClass] ?? 0,
      probabilities: icm.probabilities,
    },
    te: {
      ...data.te,
      raw_probabilities: data.te.probabilities,
      predicted_class: te.predictedClass,
      confidence: te.probabilities[te.predictedClass] ?? 0,
      probabilities: te.probabilities,
    },
  };
}

/** Build the stored assessment from a /predict_heatmap response (shared by upload and case detail). */
export function assessmentFromHeatmap(data: PredictHeatmapResponse): Extract<EmbryoMlAssessment, { predictedExp: number }> {
  return {
    assessedAt: new Date().toISOString(),
    predictedExp: data.exp.predicted_class,
    predictedIcm: data.icm.predicted_class,
    predictedTe: data.te.predicted_class,
    probabilitiesExp: data.exp.probabilities,
    probabilitiesIcm: data.icm.probabilities,
    probabilitiesTe: data.te.probabilities,
    heatmaps: {
      exp: heatmapDataUrl(data.exp.heatmap_png_base64),
      icm: heatmapDataUrl(data.icm.heatmap_png_base64),
      te: heatmapDataUrl(data.te.heatmap_png_base64),
    },
    rawProbabilities: {
      exp: data.exp.raw_probabilities ?? data.exp.probabilities,
      icm: data.icm.raw_probabilities ?? data.icm.probabilities,
      te: data.te.raw_probabilities ?? data.te.probabilities,
    },
    focus: { exp: data.exp.focus, icm: data.icm.focus, te: data.te.focus },
  };
}

/**
 * Call predict API for a File (e.g. first upload). Returns undefined if VITE_ML_API_URL is not set.
 * Uses /predict_heatmap so every new assessment also carries its Grad-CAM overlays; falls back to
 * plain /predict (no heatmaps) if the running ML API is an older version that doesn't have that
 * endpoint yet, so upgrading serve_predict_gardner.py isn't a hard requirement to keep working.
 */
export async function fetchMlAssessmentForImage(
  file: File
): Promise<EmbryoMlAssessment | undefined> {
  const base = mlApiBase();
  if (!base) return undefined;
  const assessedAt = new Date().toISOString();
  try {
    const data = await predictGardnerGradesWithHeatmap(base, file);
    return { ...assessmentFromHeatmap(data), assessedAt };
  } catch {
    // Older ML API without /predict_heatmap (404), or any other heatmap-specific failure:
    // fall back to plain grades so upload still works, just without heatmaps this time.
    try {
      const data = await predictGardnerGrades(base, file);
      return {
        assessedAt,
        predictedExp: data.predicted_exp,
        predictedIcm: data.predicted_icm,
        predictedTe: data.predicted_te,
        probabilitiesExp: data.probabilities_exp,
        probabilitiesIcm: data.probabilities_icm,
        probabilitiesTe: data.probabilities_te,
      };
    } catch (e) {
      return {
        assessedAt,
        error: e instanceof Error ? e.message : 'Prediction failed',
      };
    }
  }
}


// ---------------------------------------------------------------------------------------------
// Explanation, transfer priority and chat (all computed by the ML API from the model output)
// ---------------------------------------------------------------------------------------------
type OkAssessment = Extract<EmbryoMlAssessment, { predictedExp: number }>;

export type ExplainedHead = {
  name: string;
  predicted_class: number;
  grade_text: string;
  confidence: number;
  low_confidence: boolean;
  not_assessable: boolean;
  explanation: string;
};

export type PriorityPart = { factor: string; value: string; points: number; max?: number; reason?: string };

export type TransferPriority = {
  score: number;
  band: 'High' | 'Medium' | 'Low';
  morphology: PriorityPart[];
  clinical: PriorityPart[];
  clinical_provided: boolean;
  not_provided: string[];
  flags: string[];
  formula: string;
  disclaimer: string;
  low_confidence_penalty: number;
};

export type ExplainResponse = {
  heads: Record<'exp' | 'icm' | 'te', ExplainedHead>;
  priority: TransferPriority;
  notes: string[];
  llm_available: boolean;
};

export type OutlookResponse = {
  outlook: 'Favourable' | 'Moderate' | 'Guarded';
  band: 'High' | 'Medium' | 'Low';
  score: number;
  summary: string;
  strengths: string[];
  concerns: string[];
  discuss: string[];
  source: 'llm' | 'grounded';
  model: string | null;
  notice: string | null;
  disclaimer: string;
};

export type ChatTurn = { role: 'user' | 'assistant'; content: string };
export type ChatResponse = { answer: string; source: 'llm' | 'grounded'; model: string | null; notice: string | null };

/** camelCase form fields to the API's snake_case. */
export function toApiClinical(c: ClinicalInputs | undefined) {
  if (!c) return null;
  return {
    female_age: c.femaleAge,
    amh: c.amh ?? null,
    fsh: c.fsh ?? null,
    bmi: c.bmi ?? null,
    prior_cycles: c.priorCycles ?? null,
    failed_transfers: c.failedTransfers ?? null,
    infertility_cause: c.infertilityCause ?? null,
  };
}

/** The real model output (never the demo-scripted confidences). Falls back to displayed probabilities for older records. */
export function toApiModelOutput(ml: OkAssessment) {
  const raw = ml.rawProbabilities;
  return {
    probabilities_exp: raw?.exp ?? ml.probabilitiesExp,
    probabilities_icm: raw?.icm ?? ml.probabilitiesIcm,
    probabilities_te: raw?.te ?? ml.probabilitiesTe,
    focus_exp: ml.focus?.exp ?? null,
    focus_icm: ml.focus?.icm ?? null,
    focus_te: ml.focus?.te ?? null,
  };
}

async function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export function explainCase(baseUrl: string, ml: OkAssessment, clinical: ClinicalInputs | undefined, signal?: AbortSignal) {
  return postJson<ExplainResponse>(`${trimBase(baseUrl)}/explain`, { model_output: toApiModelOutput(ml), clinical: toApiClinical(clinical) }, signal);
}

export function chatAboutCase(baseUrl: string, ml: OkAssessment, clinical: ClinicalInputs | undefined, messages: ChatTurn[], signal?: AbortSignal) {
  return postJson<ChatResponse>(
    `${trimBase(baseUrl)}/chat`,
    { model_output: toApiModelOutput(ml), clinical: toApiClinical(clinical), messages },
    signal
  );
}

/** Estimated outlook for one embryo: label from the heuristic band, wording from the LLM (or a built-in fallback). */
export function assessCase(baseUrl: string, ml: OkAssessment, clinical: ClinicalInputs | undefined, signal?: AbortSignal) {
  return postJson<OutlookResponse>(`${trimBase(baseUrl)}/assess`, { model_output: toApiModelOutput(ml), clinical: toApiClinical(clinical) }, signal);
}
