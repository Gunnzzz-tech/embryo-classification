import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchMlAssessmentForImage, mlApiBase } from '../api/mlPredict';
import { useCases, nextCaseId, createEmbryoImage } from '../context/CasesContext';
import type { EmbryoCase, EmbryoGrade, EmbryoDay } from '../types';
import styles from './AddPatientRecord.module.css';

const EMBRYO_DAYS: EmbryoDay[] = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6'];
const GRADES: EmbryoGrade[] = ['A', 'B', 'C', 'D', 'N/A'];

interface ImageEntry {
  key: string;
  file: File | null;
  previewUrl: string | null;
  embryoNumber: number;
  captureDay: EmbryoDay;
  magnification: string;
  initialGrade: EmbryoGrade;
  blastocystStage: string;
}

function newImageEntry(key: string, embryoNumber = 1): ImageEntry {
  return {
    key,
    file: null,
    previewUrl: null,
    embryoNumber,
    captureDay: 'Day 5',
    magnification: '',
    initialGrade: 'N/A',
    blastocystStage: '',
  };
}

export function AddPatientRecord() {
  const navigate = useNavigate();
  const { cases, addCase } = useCases();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [patientRef, setPatientRef] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [cycleDate, setCycleDate] = useState('');
  const [retrievalDate, setRetrievalDate] = useState('');
  const [fertilizationMethod, setFertilizationMethod] = useState<EmbryoCase['fertilizationMethod']>('ICSI');
  const [embryoCount, setEmbryoCount] = useState(1);
  const [selectedCount, setSelectedCount] = useState(0);
  const [primaryGrade, setPrimaryGrade] = useState<EmbryoGrade>('A');
  const [secondaryGrade, setSecondaryGrade] = useState<EmbryoGrade>('A');
  const [notes, setNotes] = useState('');
  const [imageEntries, setImageEntries] = useState<ImageEntry[]>([newImageEntry('0')]);
  const [dropActive, setDropActive] = useState(false);
  const fileInputBulkRef = useRef<HTMLInputElement>(null);
  const entriesRef = useRef(imageEntries);
  entriesRef.current = imageEntries;

  useEffect(() => {
    return () => {
      entriesRef.current.forEach((e) => {
        if (e.previewUrl) URL.revokeObjectURL(e.previewUrl);
      });
    };
  }, []);

  const addImageRow = () => {
    setImageEntries((prev) => [...prev, newImageEntry(String(Date.now()), prev.length + 1)]);
  };

  const updateImageEntry = (key: string, updates: Partial<ImageEntry>) => {
    setImageEntries((prev) =>
      prev.map((e) => {
        if (e.key !== key) return e;
        if ('file' in updates && updates.file !== undefined) {
          if (e.previewUrl) URL.revokeObjectURL(e.previewUrl);
          const file = updates.file;
          if (!file) return { ...e, ...updates, file: null, previewUrl: null };
          return { ...e, ...updates, file, previewUrl: URL.createObjectURL(file) };
        }
        const { previewUrl: _p, ...rest } = updates;
        return { ...e, ...rest };
      })
    );
  };

  const removeImageRow = (key: string) => {
    setImageEntries((prev) => {
      const e = prev.find((x) => x.key === key);
      if (e?.previewUrl) URL.revokeObjectURL(e.previewUrl);
      return prev.filter((x) => x.key !== key);
    });
  };

  const appendFilesAsRows = (list: FileList | null) => {
    if (!list?.length) return;
    const imgs = Array.from(list).filter((f) => f.type.startsWith('image/'));
    if (!imgs.length) return;
    setImageEntries((prev) => {
      const start = prev.length;
      const rows = imgs.map((file, i) => ({
        ...newImageEntry(`${Date.now()}-${i}`, start + i + 1),
        file,
        previewUrl: URL.createObjectURL(file),
      }));
      return [...prev, ...rows];
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    const ref = patientRef.trim();
    if (!ref) {
      setError('Patient reference is required.');
      setSubmitting(false);
      return;
    }
    if (!cycleDate) {
      setError('Cycle date is required.');
      setSubmitting(false);
      return;
    }

    const entriesWithFile = imageEntries.filter((e) => e.file);
    const patientAgeNum = patientAge ? parseInt(patientAge, 10) : undefined;
    const embryoImagesRaw = entriesWithFile.length
      ? await Promise.all(
          entriesWithFile.map((e) =>
            createEmbryoImage(e.file!, e.embryoNumber, e.captureDay, {
              magnification: e.magnification || undefined,
              initialGrade: e.initialGrade === 'N/A' ? undefined : e.initialGrade,
              blastocystStage: e.blastocystStage || undefined,
            })
          )
        )
      : undefined;
    const embryoImages =
      embryoImagesRaw && mlApiBase()
        ? await Promise.all(
            embryoImagesRaw.map(async (img, i) => {
              const file = entriesWithFile[i]?.file;
              if (!file) return img;
              const ml = await fetchMlAssessmentForImage(file, { age: patientAgeNum });
              return ml ? { ...img, mlAssessment: ml } : img;
            })
          )
        : embryoImagesRaw;
    const caseMl = embryoImages?.[0]?.mlAssessment;

    const caseId = nextCaseId(cases);
    const id = `new-${Date.now()}`;
    const newCase: EmbryoCase = {
      id,
      caseId,
      patientRef: ref,
      cycleDate,
      embryoCount,
      selectedCount,
      primaryGrade,
      secondaryGrade,
      status: 'pending_review',
      notes: notes.trim() || undefined,
      patientAge: patientAgeNum,
      retrievalDate: retrievalDate || undefined,
      fertilizationMethod,
      embryoImages,
      ...(caseMl && { mlAssessment: caseMl }),
    };

    addCase(newCase);
    setSubmitting(false);
    navigate(`/cases/${id}`);
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1>Add patient record</h1>
        <p className={styles.subtitle}>
          Patient and cycle details, plus embryo images for fertility assessment
        </p>
      </header>

      <form onSubmit={handleSubmit} className={styles.form}>
        {error && <div className={styles.error}>{error}</div>}

        <section className={styles.section}>
          <h2>Patient & cycle</h2>
          <div className={styles.grid}>
            <label className={styles.label}>
              Patient reference <span className={styles.required}>*</span>
              <input
                type="text"
                value={patientRef}
                onChange={(e) => setPatientRef(e.target.value)}
                placeholder="PT-****1234"
                required
              />
            </label>
            <label className={styles.label}>
              Patient age
              <input
                type="number"
                min={18}
                max={55}
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                placeholder="e.g. 34"
              />
            </label>
            <label className={styles.label}>
              Cycle date <span className={styles.required}>*</span>
              <input
                type="date"
                value={cycleDate}
                onChange={(e) => setCycleDate(e.target.value)}
                required
              />
            </label>
            <label className={styles.label}>
              Retrieval date
              <input
                type="date"
                value={retrievalDate}
                onChange={(e) => setRetrievalDate(e.target.value)}
              />
            </label>
            <label className={styles.label}>
              Fertilization method
              <select
                value={fertilizationMethod ?? ''}
                onChange={(e) =>
                  setFertilizationMethod(
                    (e.target.value as EmbryoCase['fertilizationMethod']) || undefined
                  )
                }
              >
                <option value="IVF">IVF</option>
                <option value="ICSI">ICSI</option>
                <option value="IVF+ICSI">IVF + ICSI</option>
              </select>
            </label>
            <label className={styles.label}>
              Number of embryos
              <input
                type="number"
                min={0}
                value={embryoCount}
                onChange={(e) => setEmbryoCount(parseInt(e.target.value, 10) || 0)}
              />
            </label>
            <label className={styles.label}>
              Selected for transfer
              <input
                type="number"
                min={0}
                value={selectedCount}
                onChange={(e) => setSelectedCount(parseInt(e.target.value, 10) || 0)}
              />
            </label>
          </div>
        </section>

        <section className={styles.section}>
          <h2>Initial grading</h2>
          <div className={styles.gradeRow}>
            <label className={styles.label}>
              Primary grade
              <select
                value={primaryGrade}
                onChange={(e) => setPrimaryGrade(e.target.value as EmbryoGrade)}
              >
                {GRADES.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </label>
            <label className={styles.label}>
              Secondary grade
              <select
                value={secondaryGrade}
                onChange={(e) => setSecondaryGrade(e.target.value as EmbryoGrade)}
              >
                {GRADES.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </label>
          </div>
          <label className={styles.label}>
            Notes
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Optional clinical notes"
              rows={2}
            />
          </label>
        </section>

        <section className={styles.section}>
          <h2>Embryo images (for fertility detection)</h2>
          <p className={styles.hint}>
            Upload embryo microscope images. Drag and drop multiple files here, or use browse per row. Capture day, magnification, and optional grade help ensure accurate assessment.
          </p>
          <div
            className={`${styles.dropZone} ${dropActive ? styles.dropZoneActive : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDropActive(true);
            }}
            onDragLeave={() => setDropActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDropActive(false);
              appendFilesAsRows(e.dataTransfer.files);
            }}
          >
            <input
              ref={fileInputBulkRef}
              type="file"
              accept="image/*"
              multiple
              className={styles.hiddenBulkInput}
              onChange={(e) => {
                appendFilesAsRows(e.target.files);
                e.target.value = '';
              }}
            />
            <button
              type="button"
              className={styles.dropZoneBtn}
              onClick={() => fileInputBulkRef.current?.click()}
            >
              Drop images here or click to add multiple
            </button>
          </div>
          {imageEntries.map((entry) => (
            <div key={entry.key} className={styles.imageRow}>
              {entry.previewUrl && (
                <img src={entry.previewUrl} alt="" className={styles.rowThumb} />
              )}
              <div className={styles.imageUpload}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => updateImageEntry(entry.key, { file: e.target.files?.[0] ?? null })}
                />
                {entry.file && (
                  <span className={styles.fileName}>{entry.file.name}</span>
                )}
              </div>
              <label className={styles.shortLabel}>
                Embryo #
                <input
                  type="number"
                  min={1}
                  value={entry.embryoNumber}
                  onChange={(e) =>
                    updateImageEntry(entry.key, {
                      embryoNumber: parseInt(e.target.value, 10) || 1,
                    })
                  }
                />
              </label>
              <label className={styles.shortLabel}>
                Day
                <select
                  value={entry.captureDay}
                  onChange={(e) =>
                    updateImageEntry(entry.key, { captureDay: e.target.value as EmbryoDay })
                  }
                >
                  {EMBRYO_DAYS.map((d) => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </label>
              <label className={styles.shortLabel}>
                Magnification
                <input
                  type="text"
                  value={entry.magnification}
                  onChange={(e) => updateImageEntry(entry.key, { magnification: e.target.value })}
                  placeholder="e.g. 400x"
                />
              </label>
              <label className={styles.shortLabel}>
                Initial grade
                <select
                  value={entry.initialGrade}
                  onChange={(e) =>
                    updateImageEntry(entry.key, { initialGrade: e.target.value as EmbryoGrade })
                  }
                >
                  {GRADES.map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </label>
              <label className={styles.shortLabel}>
                Blastocyst (Day 5/6)
                <input
                  type="text"
                  value={entry.blastocystStage}
                  onChange={(e) =>
                    updateImageEntry(entry.key, { blastocystStage: e.target.value })
                  }
                  placeholder="e.g. 3AA"
                />
              </label>
              <button
                type="button"
                className={styles.removeBtn}
                onClick={() => removeImageRow(entry.key)}
                aria-label="Remove image"
              >
                ×
              </button>
            </div>
          ))}
          <button type="button" className={styles.addImageBtn} onClick={addImageRow}>
            + Add another embryo image
          </button>
        </section>

        <div className={styles.actions}>
          <button type="submit" className={styles.submit} disabled={submitting}>
            {submitting ? 'Saving…' : 'Save patient record'}
          </button>
          <button type="button" className={styles.cancel} onClick={() => navigate('/cases')}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}
