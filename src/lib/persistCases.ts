import { mockCases } from '../mockData';
import type { EmbryoCase } from '../types';

export const CASES_STORAGE_KEY = 'embryo-fertility-cases-v1';

export function loadCasesFromStorage(): EmbryoCase[] | null {
  try {
    const raw = localStorage.getItem(CASES_STORAGE_KEY);
    if (raw == null) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return null;
    return parsed as EmbryoCase[];
  } catch {
    return null;
  }
}

export function saveCasesToStorage(cases: EmbryoCase[]): void {
  try {
    localStorage.setItem(CASES_STORAGE_KEY, JSON.stringify(cases));
  } catch (e) {
    console.warn('Could not save cases to localStorage (size limit or private mode):', e);
  }
}

export function getInitialCases(): EmbryoCase[] {
  return loadCasesFromStorage() ?? mockCases;
}
