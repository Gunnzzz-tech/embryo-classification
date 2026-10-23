import { useEffect, useRef, useState } from 'react';
import { chatAboutCase, type ChatResponse, type ChatTurn } from '../api/mlPredict';
import type { ClinicalInputs, EmbryoMlAssessment } from '../types';
import styles from './ChatPanel.module.css';

type OkAssessment = Extract<EmbryoMlAssessment, { predictedExp: number }>;
type Shown = ChatTurn & { source?: ChatResponse['source']; model?: string | null; notice?: string | null };

const SUGGESTIONS = [
  'Why did the model give these grades?',
  'What does the heatmap show?',
  'How was the priority score built?',
  'How did age and the other inputs affect it?',
  'How confident is the model?',
];
const MAX_CHARS = 1000;

type Props = {
  baseUrl: string;
  ml: OkAssessment;
  clinical?: ClinicalInputs;
  /** Changing this resets the conversation (e.g. when switching embryo). */
  resetKey: string;
};

export function ChatPanel({ baseUrl, ml, clinical, resetKey }: Props) {
  const [turns, setTurns] = useState<Shown[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    abortRef.current?.abort();
    setTurns([]);
    setError(null);
    setBusy(false);
  }, [resetKey]);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'nearest' });
  }, [turns, busy]);

  const send = async (text: string) => {
    const content = text.trim().slice(0, MAX_CHARS);
    if (!content || busy) return;
    const next: Shown[] = [...turns, { role: 'user', content }];
    setTurns(next);
    setInput('');
    setError(null);
    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const history: ChatTurn[] = next.map(({ role, content: c }) => ({ role, content: c }));
      const res = await chatAboutCase(baseUrl, ml, clinical, history, ctrl.signal);
      setTurns([...next, { role: 'assistant', content: res.answer, source: res.source, model: res.model, notice: res.notice }]);
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : 'The assistant is unavailable.');
    } finally {
      if (!ctrl.signal.aborted) setBusy(false);
    }
  };

  return (
    <div className={styles.panel}>
      <div className={styles.head}>
        <strong>Ask about this result</strong>
        <span className={styles.sub}>Answers use only this case&apos;s grades, heatmap focus and clinical inputs.</span>
      </div>

      <div className={styles.log} role="log" aria-live="polite">
        {turns.length === 0 && (
          <div className={styles.empty}>
            <p>Try one of these, or type your own question:</p>
            <div className={styles.chips}>
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className={styles.chip} onClick={() => void send(s)} disabled={busy}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={t.role === 'user' ? styles.userRow : styles.botRow}>
            <div className={t.role === 'user' ? styles.userBubble : styles.botBubble}>
              <p>{t.content}</p>
              {t.role === 'assistant' && (
                <span className={styles.meta}>
                  {t.source === 'llm' ? `AI assistant${t.model ? ` (${t.model})` : ''}` : 'Built-in answer from case data'}
                  {t.notice ? ` · ${t.notice}` : ''}
                </span>
              )}
            </div>
          </div>
        ))}
        {busy && <div className={styles.typing}>Thinking…</div>}
        {error && <div className={styles.error}>Could not get an answer: {error}</div>}
        <div ref={endRef} />
      </div>

      <form
        className={styles.form}
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <input
          type="text"
          value={input}
          maxLength={MAX_CHARS}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask why a grade was given, what the heatmap shows, how the score works…"
          aria-label="Question about this result"
          disabled={busy}
        />
        <button type="submit" disabled={busy || input.trim() === ''}>
          Send
        </button>
      </form>
      <p className={styles.footnote}>
        Explains the tool&apos;s output only. It cannot predict pregnancy or recommend treatment; that decision belongs to the clinical team.
      </p>
    </div>
  );
}
