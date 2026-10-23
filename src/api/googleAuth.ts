/**
 * Google "Sign in with Google" (Google Identity Services), client-only.
 *
 * IMPORTANT: this app has no backend, so the ID token below is decoded but never
 * verified against Google's servers. That's fine for this demo (the existing
 * doctor-ID sign-in is equally unverified) but is NOT secure — anyone could
 * hand-craft a similar-looking token. Do not rely on this for real access control.
 */

export type GoogleProfile = {
  name?: string;
  email?: string;
  picture?: string;
};

type CredentialResponse = { credential: string };

interface GoogleIdApi {
  initialize(config: { client_id: string; callback: (resp: CredentialResponse) => void }): void;
  renderButton(parent: HTMLElement, options: Record<string, unknown>): void;
}

declare global {
  interface Window {
    google?: { accounts: { id: GoogleIdApi } };
  }
}

const SCRIPT_SRC = 'https://accounts.google.com/gsi/client';
let loadPromise: Promise<void> | undefined;

/** Load the Google Identity Services script once (safe to call repeatedly). */
export function loadGoogleIdentityScript(): Promise<void> {
  if (window.google?.accounts?.id) return Promise.resolve();
  if (loadPromise) return loadPromise;

  loadPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${SCRIPT_SRC}"]`);
    if (existing) {
      existing.addEventListener('load', () => resolve());
      existing.addEventListener('error', () => reject(new Error('Failed to load Google sign-in script')));
      return;
    }
    const script = document.createElement('script');
    script.src = SCRIPT_SRC;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Google sign-in script'));
    document.head.appendChild(script);
  });

  return loadPromise;
}

/** Decode a Google ID token's payload WITHOUT verifying its signature. See file note above. */
export function decodeGoogleCredential(credential: string): GoogleProfile {
  const payload = credential.split('.')[1];
  if (!payload) return {};
  try {
    const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    const claims = JSON.parse(decodeURIComponent(escape(json))) as GoogleProfile;
    return claims;
  } catch {
    return {};
  }
}

export function googleClientId(): string | undefined {
  const id = import.meta.env.VITE_GOOGLE_CLIENT_ID;
  return typeof id === 'string' && id.trim() ? id.trim() : undefined;
}

/**
 * Initialize Google Identity Services and render the "Sign in with Google" button
 * into `container`. Calls `onProfile` with the decoded (unverified) profile once
 * the user completes sign-in.
 */
export async function renderGoogleSignInButton(
  container: HTMLElement,
  clientId: string,
  onProfile: (profile: GoogleProfile) => void
): Promise<void> {
  await loadGoogleIdentityScript();
  const id = window.google?.accounts?.id;
  if (!id) throw new Error('Google sign-in script did not load correctly');

  id.initialize({
    client_id: clientId,
    callback: (resp) => onProfile(decodeGoogleCredential(resp.credential)),
  });
  id.renderButton(container, { theme: 'outline', size: 'large', width: 320, text: 'signin_with' });
}
