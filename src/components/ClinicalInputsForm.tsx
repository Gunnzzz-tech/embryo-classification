import type { ClinicalInputs, InfertilityCause } from '../types';
import styles from './ClinicalInputsForm.module.css';

/** Form state is kept as strings so the user can type freely; parseClinical() validates and converts. */
export type ClinicalFormValues = {
  femaleAge: string;
  amh: string;
  fsh: string;
  bmi: string;
  priorCycles: string;
  failedTransfers: string;
  infertilityCause: '' | InfertilityCause;
};

export const emptyClinicalForm: ClinicalFormValues = {
  femaleAge: '',
  amh: '',
  fsh: '',
  bmi: '',
  priorCycles: '',
  failedTransfers: '',
  infertilityCause: '',
};

export function clinicalToForm(c: ClinicalInputs | undefined): ClinicalFormValues {
  if (!c) return emptyClinicalForm;
  const s = (n?: number) => (n == null ? '' : String(n));
  return {
    femaleAge: s(c.femaleAge),
    amh: s(c.amh),
    fsh: s(c.fsh),
    bmi: s(c.bmi),
    priorCycles: s(c.priorCycles),
    failedTransfers: s(c.failedTransfers),
    infertilityCause: c.infertilityCause ?? '',
  };
}

const CAUSES: Array<{ value: InfertilityCause; label: string }> = [
  { value: 'unexplained', label: 'Unexplained' },
  { value: 'male_factor', label: 'Male factor' },
  { value: 'tubal', label: 'Tubal factor' },
  { value: 'ovulatory', label: 'Ovulatory disorder (e.g. PCOS)' },
  { value: 'endometriosis', label: 'Endometriosis' },
  { value: 'diminished_reserve', label: 'Diminished ovarian reserve' },
  { value: 'uterine', label: 'Uterine factor' },
  { value: 'other', label: 'Other' },
];

type Spec = { key: keyof ClinicalFormValues; label: string; unit: string; min: number; max: number; integer: boolean; required?: boolean; hint: string };

// Ranges mirror the ML API's validation (explain.ClinicalInputs), so a value that passes here is accepted there.
const FIELDS: Spec[] = [
  { key: 'femaleAge', label: 'Female age', unit: 'years', min: 18, max: 55, integer: true, required: true, hint: 'Age at egg retrieval. The strongest single factor.' },
  { key: 'amh', label: 'AMH', unit: 'ng/mL', min: 0, max: 30, integer: false, hint: 'Ovarian reserve marker. Leave blank if not measured.' },
  { key: 'fsh', label: 'Basal FSH (day 2-3)', unit: 'mIU/mL', min: 0, max: 60, integer: false, hint: 'Leave blank if not measured.' },
  { key: 'bmi', label: 'BMI', unit: 'kg/m²', min: 12, max: 60, integer: false, hint: 'Weight (kg) divided by height (m) squared.' },
  { key: 'priorCycles', label: 'Previous IVF/ICSI cycles', unit: 'count', min: 0, max: 20, integer: true, hint: 'Completed stimulation cycles before this one.' },
  { key: 'failedTransfers', label: 'Previous failed transfers', unit: 'count', min: 0, max: 15, integer: true, hint: 'Embryo transfers that did not lead to a pregnancy.' },
];

/** Validate and convert. Out-of-range values are errors, never silently clipped. */
export function parseClinical(v: ClinicalFormValues): { value?: ClinicalInputs; errors: Partial<Record<keyof ClinicalFormValues, string>> } {
  const errors: Partial<Record<keyof ClinicalFormValues, string>> = {};
  const nums: Partial<Record<keyof ClinicalFormValues, number>> = {};
  for (const f of FIELDS) {
    const raw = v[f.key].trim();
    if (raw === '') {
      if (f.required) errors[f.key] = `${f.label} is required.`;
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) errors[f.key] = 'Enter a number.';
    else if (f.integer && !Number.isInteger(n)) errors[f.key] = 'Enter a whole number.';
    else if (n < f.min || n > f.max) errors[f.key] = `Must be between ${f.min} and ${f.max} ${f.unit}.`;
    else nums[f.key] = n;
  }
  if (Object.keys(errors).length > 0) return { errors };
  return {
    errors,
    value: {
      femaleAge: nums.femaleAge as number,
      amh: nums.amh,
      fsh: nums.fsh,
      bmi: nums.bmi,
      priorCycles: nums.priorCycles,
      failedTransfers: nums.failedTransfers,
      infertilityCause: v.infertilityCause || undefined,
    },
  };
}

type Props = {
  values: ClinicalFormValues;
  onChange: (next: ClinicalFormValues) => void;
  errors?: Partial<Record<keyof ClinicalFormValues, string>>;
};

export function ClinicalInputsForm({ values, onChange, errors = {} }: Props) {
  const set = (key: keyof ClinicalFormValues, val: string) => onChange({ ...values, [key]: val } as ClinicalFormValues);
  return (
    <fieldset className={styles.fieldset}>
      <legend className={styles.legend}>Clinical details</legend>
      <p className={styles.intro}>
        Doctors read an embryo image together with these factors. They do not change the image grades; they feed the
        transfer-priority score and the explanations. Only age is required.
      </p>
      <div className={styles.grid}>
        {FIELDS.map((f) => (
          <label key={f.key} className={styles.field}>
            <span className={styles.label}>
              {f.label} <span className={styles.unit}>({f.unit})</span>
              {f.required && <span className={styles.req}> *</span>}
            </span>
            <input
              type="number"
              inputMode="decimal"
              step={f.integer ? 1 : 'any'}
              min={f.min}
              max={f.max}
              value={values[f.key]}
              onChange={(e) => set(f.key, e.target.value)}
              aria-invalid={errors[f.key] ? true : undefined}
              required={f.required}
            />
            {errors[f.key] ? <span className={styles.err}>{errors[f.key]}</span> : <span className={styles.hint}>{f.hint}</span>}
          </label>
        ))}
        <label className={styles.field}>
          <span className={styles.label}>Main infertility cause</span>
          <select value={values.infertilityCause} onChange={(e) => set('infertilityCause', e.target.value)}>
            <option value="">Not specified</option>
            {CAUSES.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
          <span className={styles.hint}>As recorded in the patient&apos;s work-up.</span>
        </label>
      </div>
    </fieldset>
  );
}
