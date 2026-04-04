import { createContext, useCallback, useContext, useMemo, useState } from 'react';

const AUTH_KEY = 'embryo-admin-auth';

type AuthState = {
  doctorId: string;
  signedIn: true;
} | { signedIn: false };

function loadStored(): AuthState {
  try {
    const raw = localStorage.getItem(AUTH_KEY);
    if (!raw) return { signedIn: false };
    const { doctorId } = JSON.parse(raw) as { doctorId: string };
    if (doctorId) return { doctorId, signedIn: true };
  } catch {
    // ignore
  }
  return { signedIn: false };
}

type AuthContextValue = {
  auth: AuthState;
  signIn: (doctorId: string, _password: string) => boolean;
  signOut: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [auth, setAuth] = useState<AuthState>(loadStored);

  const signIn = useCallback((doctorId: string, _password: string) => {
    const id = doctorId.trim();
    // For now: any non-empty ID and any password verifies
    if (!id) return false;
    const next: AuthState = { doctorId: id, signedIn: true };
    setAuth(next);
    localStorage.setItem(AUTH_KEY, JSON.stringify({ doctorId: id }));
    return true;
  }, []);

  const signOut = useCallback(() => {
    setAuth({ signedIn: false });
    localStorage.removeItem(AUTH_KEY);
  }, []);

  const value = useMemo(
    () => ({ auth, signIn, signOut }),
    [auth, signIn, signOut]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
