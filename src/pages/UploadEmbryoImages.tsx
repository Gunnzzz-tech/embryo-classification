import { useState, useRef, useCallback, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { fetchMlAssessmentForImage, mlApiBase } from '../api/mlPredict';
import { useCases, nextCaseId, createEmbryoImage } from '../context/CasesContext';
import type { EmbryoCase } from '../types';
import styles from './UploadEmbryoImages.module.css';

type FileItem = { id: string; file: File; preview: string };

export function UploadEmbryoImages() {
  const navigate = useNavigate();
  const { cases, addCase } = useCases();
  const [items, setItems] = useState<FileItem[]>([]);
  const [patientRef, setPatientRef] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const previewsToRevoke = useRef<string[]>([]);

  const revokeAll = useCallback(() => {
    previewsToRevoke.current.forEach((u) => URL.revokeObjectURL(u));
    previewsToRevoke.current = [];
  }, []);

  useEffect(() => () => revokeAll(), [revokeAll]);

  const addFiles = useCallback(
    (list: FileList | File[] | null) => {
      if (!list || list.length === 0) return;
      const arr = Array.from(list).filter((f) => f.type.startsWith('image/'));
      const next: FileItem[] = arr.map((file) => {
        const preview = URL.createObjectURL(file);
        previewsToRevoke.current.push(preview);
        return { id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, file, preview };
      });
      setItems((prev) => [...prev, ...next]);
    },
    []
  );

  const removeItem = (id: string) => {
    setItems((prev) => {
      const it = prev.find((x) => x.id === id);
      if (it) {
        URL.revokeObjectURL(it.preview);
        previewsToRevoke.current = previewsToRevoke.current.filter((u) => u !== it.preview);
      }
      return prev.filter((x) => x.id !== id);
    });
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    addFiles(e.dataTransfer.files);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = () => setDragActive(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (items.length === 0) {
      setError('Add at least one image (drag & drop or browse).');
      return;
    }
    setSubmitting(true);
    const ref = patientRef.trim() || `PT-UPLOAD-${new Date().toISOString().slice(0, 10)}`;
    const cycleDate = new Date().toISOString().slice(0, 10);
    const embryoImages = await Promise.all(
      items.map((it, i) =>
        createEmbryoImage(it.file, i + 1, 'Day 5', {})
      )
    );
    const base = mlApiBase();
    const mlList = base
      ? await Promise.all(items.map((it) => fetchMlAssessmentForImage(it.file)))
      : [];
    const withMl: typeof embryoImages = embryoImages.map((img, i) =>
      mlList[i] ? { ...img, mlAssessment: mlList[i] } : img
    );
    const caseMl = withMl[0]?.mlAssessment;
    const caseId = nextCaseId(cases);
    const id = `new-${Date.now()}`;
    const newCase: EmbryoCase = {
      id,
      caseId,
      patientRef: ref,
      cycleDate,
      embryoCount: items.length,
      selectedCount: 0,
      primaryGrade: 'N/A',
      secondaryGrade: 'N/A',
      status: 'pending_review',
      notes: 'Created from image upload.',
      embryoImages: withMl,
      ...(caseMl && { mlAssessment: caseMl }),
    };
    addCase(newCase);
    revokeAll();
    setItems([]);
    setSubmitting(false);
    navigate(`/cases/${id}`);
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Upload embryo images</h1>
        <p className={styles.subtitle}>
          Drag and drop microscope images here or browse. Multiple files supported. Cases (including LB scores when{' '}
          <code className={styles.codeInline}>VITE_ML_API_URL</code> is set) are saved in this browser via localStorage.
        </p>
        <Link to="/add-patient" className={styles.linkFull}>
          Need full patient &amp; cycle form? → Add patient record
        </Link>
      </header>

      <form onSubmit={handleSubmit} className={styles.form}>
        {error && <div className={styles.error}>{error}</div>}

        <label className={styles.label}>
          Patient reference (optional)
          <input
            type="text"
            value={patientRef}
            onChange={(e) => setPatientRef(e.target.value)}
            placeholder="Leave blank to auto-generate PT-UPLOAD-YYYY-MM-DD"
          />
        </label>

        <div
          className={`${styles.dropZone} ${dragActive ? styles.dropZoneActive : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            multiple
            className={styles.hiddenInput}
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = '';
            }}
          />
          <span className={styles.dropIcon}>⬆</span>
          <p className={styles.dropTitle}>Drop embryo images here</p>
          <p className={styles.dropHint}>or click to browse — PNG, JPG, WebP</p>
        </div>

        {items.length > 0 && (
          <section className={styles.previewSection}>
            <h2>Selected ({items.length})</h2>
            <ul className={styles.previewGrid}>
              {items.map((it) => (
                <li key={it.id} className={styles.previewCard}>
                  <img src={it.preview} alt="" className={styles.previewImg} />
                  <span className={styles.previewName}>{it.file.name}</span>
                  <button
                    type="button"
                    className={styles.removeBtn}
                    onClick={(e) => {
                      e.stopPropagation();
                      removeItem(it.id);
                    }}
                    aria-label="Remove"
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <div className={styles.actions}>
          <button type="submit" className={styles.submit} disabled={submitting || items.length === 0}>
            {submitting
              ? mlApiBase()
                ? 'Creating case & LB scores…'
                : 'Saving…'
              : 'Create case from images'}
          </button>
          <Link to="/cases" className={styles.cancel}>
            Cancel
          </Link>
        </div>
      </form>
    </div>
  );
}
