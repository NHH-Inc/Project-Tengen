// The team key a hosted Tengen asks for (ingest/hosted.py). Kept in this browser only: it is a
// per-viewer convenience, and a private window or cleared storage simply asks for it again.

const STORAGE = 'tengen-team-key';

export function getTeamKey(): string {
  try {
    return localStorage.getItem(STORAGE) ?? '';
  } catch {
    return '';
  }
}

export function setTeamKey(key: string): void {
  try {
    if (key) localStorage.setItem(STORAGE, key);
    else localStorage.removeItem(STORAGE);
  } catch {
    // Storage unavailable: the key lasts until the page reloads, which is still usable.
  }
  memory = key;
}

let memory = '';

/** The key to send: stored, or remembered for this page if storage is blocked. */
export function currentKey(): string {
  return getTeamKey() || memory;
}

/** Media elements cannot send headers, so their URLs carry the key as a query parameter. */
export function withKey(url: string): string {
  const key = currentKey();
  if (!key) return url;
  return `${url}${url.includes('?') ? '&' : '?'}key=${encodeURIComponent(key)}`;
}
