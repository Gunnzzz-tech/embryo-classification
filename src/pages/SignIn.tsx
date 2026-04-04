import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import styles from './SignIn.module.css';

export function SignIn() {
  const [doctorId, setDoctorId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const from = (location.state as { from?: { pathname: string } } | null)?.from?.pathname ?? '/';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (signIn(doctorId, password)) {
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
          <p className={styles.tagline}>Sign in with your doctor credentials</p>
        </div>
        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className={styles.error}>{error}</div>}
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
        <p className={styles.note}>For now, any doctor ID and password will sign you in.</p>
      </div>
    </div>
  );
}
