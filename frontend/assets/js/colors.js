/**
 * Stable per-class colours.
 *
 * The same rolling hash is implemented in `app/media/drawing.py`, so a class gets
 * the same colour on the browser canvas and in a server-rendered video.
 */

const cache = new Map();

/** @returns {string} an `hsl(...)` colour, stable for a given class name. */
export function colorFor(name) {
  const key = String(name);
  const hit = cache.get(key);
  if (hit) return hit;

  let hue = 0;
  for (const char of key) hue = (hue * 31 + char.charCodeAt(0)) % 360;
  const color = `hsl(${hue}, 90%, 60%)`;
  cache.set(key, color);
  return color;
}

/** Same colour at a given alpha, for mask fills. */
export function colorForAlpha(name, alpha) {
  return colorFor(name).replace("hsl", "hsla").replace(")", `, ${alpha})`);
}
