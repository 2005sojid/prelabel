/**
 * A minimal store-only ZIP writer — no dependencies, no compression.
 *
 * Images are already compressed, so deflating them would cost CPU for nothing.
 *
 * Three correctness details that a naive writer gets wrong, and that fail
 * *silently* — producing an archive that looks fine until someone opens it:
 *
 *  - **Unicode names.** Without the UTF-8 flag (general-purpose bit 11), a name
 *    like `写真.jpg` is interpreted as the reader's local codepage and comes out
 *    as mojibake. We set the flag and encode names as UTF-8.
 *  - **Duplicate names.** Two files called `img.jpg` from different folders
 *    produce two entries with the same path; most tools silently keep one. Names
 *    are made unique instead.
 *  - **The 4 GB ceiling.** Sizes and offsets in the classic format are 32-bit.
 *    Past that a writer has to switch to ZIP64 or it emits a corrupt archive. We
 *    refuse with an explanation rather than hand over broken output.
 */

/** Largest value the 32-bit size and offset fields can hold. */
export const MAX_ZIP_BYTES = 0xffffffff;

/** Entry counts in the end-of-central-directory record are 16-bit. */
export const MAX_ZIP_ENTRIES = 0xffff;

/** Bit 11: filenames and comments are UTF-8. */
const FLAG_UTF8 = 0x0800;

const SIGNATURE_LOCAL = 0x04034b50;
const SIGNATURE_CENTRAL = 0x02014b50;
const SIGNATURE_EOCD = 0x06054b50;

const LOCAL_HEADER_BYTES = 30;
const CENTRAL_HEADER_BYTES = 46;
const EOCD_BYTES = 22;

const VERSION = 20; // 2.0 — the minimum that understands directories

export class ZipLimitError extends Error {
  constructor(message) {
    super(message);
    this.name = "ZipLimitError";
  }
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** MS-DOS time/date, as the format requires. */
function dosStamp(date = new Date()) {
  const year = Math.max(1980, date.getFullYear());
  return {
    time: (date.getHours() << 11) | (date.getMinutes() << 5) | (Math.floor(date.getSeconds() / 2) & 0x1f),
    date: ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate(),
  };
}

/**
 * Make every path unique by appending ` (2)`, ` (3)`, … before the extension.
 * Comparison is case-insensitive because Windows and macOS treat it that way.
 */
export function uniquifyNames(entries) {
  const seen = new Map();
  return entries.map((entry) => {
    const key = entry.name.toLowerCase();
    const count = seen.get(key) || 0;
    seen.set(key, count + 1);
    if (count === 0) return entry;

    const dot = entry.name.lastIndexOf(".");
    const stem = dot > 0 ? entry.name.slice(0, dot) : entry.name;
    const suffix = dot > 0 ? entry.name.slice(dot) : "";
    return { ...entry, name: `${stem} (${count + 1})${suffix}` };
  });
}

/**
 * Build a ZIP blob.
 *
 * @param {Array<{name: string, data: Uint8Array}>} entries
 * @returns {Blob}
 * @throws {ZipLimitError} when the archive would exceed what this format can address
 */
export function makeZip(entries) {
  const files = uniquifyNames(entries);
  assertWithinLimits(files);

  const encoder = new TextEncoder();
  const stamp = dosStamp();
  const body = [];
  const central = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const crc = crc32(file.data);
    const size = file.data.length;

    const local = new DataView(new ArrayBuffer(LOCAL_HEADER_BYTES));
    local.setUint32(0, SIGNATURE_LOCAL, true);
    local.setUint16(4, VERSION, true);
    local.setUint16(6, FLAG_UTF8, true);
    local.setUint16(8, 0, true); // stored
    local.setUint16(10, stamp.time, true);
    local.setUint16(12, stamp.date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, size, true);
    local.setUint32(22, size, true);
    local.setUint16(26, nameBytes.length, true);
    body.push(new Uint8Array(local.buffer), nameBytes, file.data);

    const directory = new DataView(new ArrayBuffer(CENTRAL_HEADER_BYTES));
    directory.setUint32(0, SIGNATURE_CENTRAL, true);
    directory.setUint16(4, VERSION, true);
    directory.setUint16(6, VERSION, true);
    directory.setUint16(8, FLAG_UTF8, true);
    directory.setUint16(10, 0, true);
    directory.setUint16(12, stamp.time, true);
    directory.setUint16(14, stamp.date, true);
    directory.setUint32(16, crc, true);
    directory.setUint32(20, size, true);
    directory.setUint32(24, size, true);
    directory.setUint16(28, nameBytes.length, true);
    directory.setUint32(42, offset, true);
    central.push(new Uint8Array(directory.buffer), nameBytes);

    offset += LOCAL_HEADER_BYTES + nameBytes.length + size;
  }

  const centralSize = central.reduce((total, part) => total + part.length, 0);
  const end = new DataView(new ArrayBuffer(EOCD_BYTES));
  end.setUint32(0, SIGNATURE_EOCD, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);

  return new Blob([...body, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
}

function assertWithinLimits(files) {
  if (files.length > MAX_ZIP_ENTRIES) {
    throw new ZipLimitError(
      `This archive would hold ${files.length} files; the ZIP format used here supports ` +
        `${MAX_ZIP_ENTRIES}. Export in smaller batches.`,
    );
  }

  const encoder = new TextEncoder();
  let total = EOCD_BYTES;
  for (const file of files) {
    const nameLength = encoder.encode(file.name).length;
    total += LOCAL_HEADER_BYTES + CENTRAL_HEADER_BYTES + nameLength * 2 + file.data.length;
  }
  if (total > MAX_ZIP_BYTES) {
    const gigabytes = (total / 1024 ** 3).toFixed(1);
    throw new ZipLimitError(
      `This archive would be ${gigabytes} GB; the ZIP format used here tops out at 4 GB. ` +
        "Export in smaller batches.",
    );
  }
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function saveZip(entries, filename) {
  downloadBlob(makeZip(entries), filename);
}
