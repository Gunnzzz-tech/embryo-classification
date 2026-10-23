import { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth, type Role } from '../context/AuthContext';
import { googleClientId, renderGoogleSignInButton, type GoogleProfile } from '../api/googleAuth';
import styles from './SignIn.module.css';

/** Official Google "G" mark, so the fallback (not-yet-configured) button still reads as
 * "Sign in with Google" rather than a bare text link. */
function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true" focusable="false">
      <path
        fill="#EA4335"
        d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
      />
      <path
        fill="#4285F4"
        d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.9-2.26 5.36-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"
      />
      <path
        fill="#FBBC05"
        d="M10.53 28.59A14.5 14.5 0 019.5 24c0-1.59.27-3.13.77-4.59l-7.98-6.19A23.94 23.94 0 000 24c0 3.85.92 7.49 2.56 10.78l7.97-6.19z"
      />
      <path
        fill="#34A853"
        d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.97 6.19C6.51 42.62 14.62 48 24 48z"
      />
    </svg>
  );
}

export function SignIn() {
  const [doctorId, setDoctorId] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<Role>('nurse');
  const [error, setError] = useState('');
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: { pathname: string } } | null)?.from?.pathname ?? '/';

  // Kept in sync with `role` state; read from inside the Google callback so it
  // always sees the currently-selected role instead of a stale closure value.
  const roleRef = useRef(role);
  useEffect(() => {
    roleRef.current = role;
  }, [role]);

  const googleContainerRef = useRef<HTMLDivElement>(null);
  const clientId = googleClientId();

  useEffect(() => {
    if (!clientId || !googleContainerRef.current) return;
    let cancelled = false;

    const handleProfile = (profile: GoogleProfile) => {
      const id = profile.email ?? profile.name;
      if (!id) {
        setError('Google sign-in did not return a name or email.');
        return;
      }
      if (signIn(id, 'google-oauth', roleRef.current)) {
        navigate(from, { replace: true });
      }
    };

    renderGoogleSignInButton(googleContainerRef.current, clientId, handleProfile).catch(() => {
      if (!cancelled) setError('Could not load Google sign-in.');
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clientId]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (signIn(doctorId, password, role)) {
      navigate(from, { replace: true });
    } else {
      setError('Please enter a doctor ID.');
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.brand}>
          <span className={styles.logo}>◈</span>
          <h1>Embryo Fertility Admin</h1>
          <p className={styles.tagline}>Sign in with your credentials</p>
        </div>
        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className={styles.error}>{error}</div>}
          <div className={styles.roleSwitch}>
            <button
              type="button"
              className={role === 'nurse' ? `${styles.roleBtn} ${styles.roleBtnActive}` : styles.roleBtn}
              onClick={() => setRole('nurse')}
            >
              Nurse
            </button>
            <button
              type="button"
              className={role === 'admin' ? `${styles.roleBtn} ${styles.roleBtnActive}` : styles.roleBtn}
              onClick={() => setRole('admin')}
            >
              Admin
            </button>
          </div>
          <label className={styles.label}>
            Doctor ID
            <input
              type="text"
              value={doctorId}
              onChange={(e) => setDoctorId(e.target.value)}
              placeholder="Enter your doctor ID"
              autoComplete="username"
              autoFocus
            />
          </label>
          <label className={styles.label}>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
            />
          </label>
          <button type="submit" className={styles.submit}>
            Sign in
          </button>
        </form>

        <div className={styles.divider}>
          <span>or</span>
        </div>

        {clientId ? (
          <div ref={googleContainerRef} className={styles.googleBtn} />
        ) : (
          <details className={styles.googleSetup}>
            <summary className={styles.googleIconBtn}>
              <GoogleIcon />
              Sign in with Google
            </summary>
            <p className={styles.googleSetupHint}>
              Not wired up yet — click to see the 4 steps to enable real Google sign-in.
            </p>
            <ol>
              <li>
                In{' '}
                <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noreferrer">
                  Google Cloud Console → Credentials
                </a>
                , create an OAuth 2.0 <strong>Web application</strong> Client ID.
              </li>
              <li>
                Add <code>http://localhost:5173</code> (your dev URL) under{' '}
                <strong>Authorized JavaScript origins</strong>.
              </li>
              <li>
                Project root: add to <code>.env.local</code>:
                <br />
                <code>VITE_GOOGLE_CLIENT_ID=&lt;your client id&gt;.apps.googleusercontent.com</code>
              </li>
              <li>
                Restart <code>npm run dev</code>.
              </li>
            </ol>
          </details>
        )}

        <p className={styles.note}>
          For now, any doctor ID and password will sign you in. Google sign-in is demo-only here too — this app
          has no backend to verify the token, so it should not be used for real access control.
        </p>
      </div>
    </div>
  );
}
