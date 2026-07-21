/**
 * ZIP writer unit tests.
 *
 * Run with: `npm test`
 *
 * The failures worth testing for here are the silent ones — an archive that
 * *looks* fine and is subtly wrong: mojibake filenames, entries that overwrite
 * each other, or 32-bit fields that wrap past 4 GB.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_ZIP_ENTRIES,
  ZipLimitError,
  crc32,
  makeZip,
  uniquifyNames,
} from "../../frontend/assets/js/export/zip.js";

const bytes = (text) => new TextEncoder().encode(text);
const entry = (name, text = "content") => ({ name, data: bytes(text) });

async function zipBytes(entries) {
  return new Uint8Array(await makeZip(entries).arrayBuffer());
}

function readUint16(data, offset) {
  return data[offset] | (data[offset + 1] << 8);
}

// --- crc32 ------------------------------------------------------------------

test("crc32 matches the reference value", () => {
  // The canonical check value for "123456789".
  assert.equal(crc32(bytes("123456789")), 0xcbf43926);
});

test("crc32 of empty input is zero", () => {
  assert.equal(crc32(new Uint8Array(0)), 0);
});

// --- name collisions --------------------------------------------------------

test("duplicate names are made unique instead of overwriting", () => {
  const result = uniquifyNames([entry("img.jpg"), entry("img.jpg"), entry("img.jpg")]);
  assert.deepEqual(result.map((item) => item.name), ["img.jpg", "img (2).jpg", "img (3).jpg"]);
});

test("uniquifying is case-insensitive, as Windows and macOS are", () => {
  const result = uniquifyNames([entry("Photo.JPG"), entry("photo.jpg")]);
  assert.equal(result[1].name, "photo (2).jpg");
});

test("names without an extension still get a suffix", () => {
  const result = uniquifyNames([entry("LICENSE"), entry("LICENSE")]);
  assert.equal(result[1].name, "LICENSE (2)");
});

test("distinct names are left alone", () => {
  const names = ["a.jpg", "b.jpg", "sub/a.jpg"];
  assert.deepEqual(uniquifyNames(names.map((n) => entry(n))).map((i) => i.name), names);
});

// --- structure --------------------------------------------------------------

test("archive carries the correct signatures and entry count", async () => {
  const data = await zipBytes([entry("a.txt"), entry("b.txt")]);

  // Local file header signature at the very start.
  assert.deepEqual([...data.slice(0, 4)], [0x50, 0x4b, 0x03, 0x04]);
  // End-of-central-directory at the end, with both entry counts.
  const eocd = data.length - 22;
  assert.deepEqual([...data.slice(eocd, eocd + 4)], [0x50, 0x4b, 0x05, 0x06]);
  assert.equal(readUint16(data, eocd + 8), 2);
  assert.equal(readUint16(data, eocd + 10), 2);
});

test("the UTF-8 flag is set so non-Latin names survive", async () => {
  const data = await zipBytes([entry("写真.jpg")]);
  const flags = readUint16(data, 6); // general-purpose bit flag, local header
  assert.equal(flags & 0x0800, 0x0800, "bit 11 (UTF-8) must be set");
});

test("an empty archive is still a valid one", async () => {
  const data = await zipBytes([]);
  assert.equal(data.length, 22); // EOCD only
  assert.equal(readUint16(data, 8), 0);
});

test("file content is stored verbatim", async () => {
  const payload = "hello world";
  const data = await zipBytes([entry("a.txt", payload)]);
  const text = new TextDecoder().decode(data);
  assert.ok(text.includes(payload), "stored entries must not be transformed");
});

// --- limits -----------------------------------------------------------------

test("refuses an archive that would exceed 4 GB rather than corrupting it", () => {
  // A fake entry that reports a huge length without allocating one.
  const huge = { name: "big.bin", data: { length: 5 * 1024 ** 3 } };
  assert.throws(() => makeZip([huge]), ZipLimitError);
});

test("the size error explains what to do", () => {
  const huge = { name: "big.bin", data: { length: 5 * 1024 ** 3 } };
  assert.throws(() => makeZip([huge]), (error) => /smaller batches/.test(error.message));
});

test("refuses more entries than the format can index", () => {
  const many = Array.from({ length: MAX_ZIP_ENTRIES + 1 }, (_, i) => ({
    name: `f${i}.txt`,
    data: new Uint8Array(0),
  }));
  assert.throws(() => makeZip(many), ZipLimitError);
});

test("an archive just under the entry limit is accepted", () => {
  const many = Array.from({ length: 10 }, (_, i) => entry(`f${i}.txt`));
  assert.ok(makeZip(many));
});
