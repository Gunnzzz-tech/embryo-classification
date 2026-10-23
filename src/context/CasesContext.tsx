import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { getInitialCases, saveCasesToStorage } from '../lib/persistCases';
import type { EmbryoCase, EmbryoImage, EmbryoDay } from '../types';

type CasesContextValue = {
  cases: EmbryoCase[];
  addCase: (case_: EmbryoCase) => void;
  updateCase: (id: string, patch: Partial<EmbryoCase>) => void;
  getCaseById: (id: string) => EmbryoCase | undefined;
  clearAllCases: () => void;
};

const CasesContext = createContext<CasesContextValue | null>(null);

export function CasesProvider({ children }: { children: React.ReactNode }) {
  const [cases, setCases] = useState<EmbryoCase[]>(getInitialCases);

  useEffect(() => {
    saveCasesToStorage(cases);
  }, [cases]);

  const addCase = useCallback((case_: EmbryoCase) => {
    setCases((prev) => [...prev, case_]);
  }, []);

  const updateCase = useCallback((id: string, patch: Partial<EmbryoCase>) => {
    setCases((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }, []);

  const getCaseById = useCallback((id: string) => {
    return cases.find((c) => c.id === id);
  }, [cases]);

  /** Wipe every patient/case record (e.g. before a demo). Does not touch sign-in. */
  const clearAllCases = useCallback(() => {
    setCases([]);
  }, []);

  const value = useMemo(
    () => ({ cases, addCase, updateCase, getCaseById, clearAllCases }),
    [cases, addCase, updateCase, getCaseById, clearAllCases]
  );

  return <CasesContext.Provider value={value}>{children}</CasesContext.Provider>;
}

export function useCases() {
  const ctx = useContext(CasesContext);
  if (!ctx) throw new Error('useCases must be used within CasesProvider');
  return ctx;
}

/** Generate next case ID (e.g. EF-2025-0846) */
export function nextCaseId(cases: EmbryoCase[]): string {
  const year = new Date().getFullYear();
  const prefix = `EF-${year}-`;
  const numbers = cases
    .filter((c) => c.caseId.startsWith(prefix))
    .map((c) => parseInt(c.caseId.slice(prefix.length), 10))
    .filter((n) => !Number.isNaN(n));
  const next = numbers.length > 0 ? Math.max(...numbers) + 1 : 1;
  return `${prefix}${String(next).padStart(4, '0')}`;
}

/** Create EmbryoImage from file and form fields */
export function createEmbryoImage(
  file: File,
  embryoNumber: number,
  captureDay: EmbryoDay,
  opts: { magnification?: string; initialGrade?: EmbryoCase['primaryGrade']; blastocystStage?: string }
): Promise<EmbryoImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      resolve({
        id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        dataUrl: reader.result as string,
        embryoNumber,
        captureDay,
        capturedAt: new Date().toISOString(),
        magnification: opts.magnification,
        initialGrade: opts.initialGrade,
        blastocystStage: opts.blastocystStage,
      });
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
