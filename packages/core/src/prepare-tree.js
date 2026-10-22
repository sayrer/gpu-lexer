import {
  TREE_FEATURE_STRIDE, TREE_PART_NEWLINE, TREE_PART_SPACE, TREE_PART_SYMBOL, TREE_PART_WORD,
} from "./constants.js";

const INITIAL_CAPACITY = 512;
const MAX_POOLED_WORKSPACES = 4;
const pool = [];
const WORKSPACE_CAPACITY = 0;
const WORKSPACE_DATA = 1;
const WORKSPACE_RANGES = 2;
const PREPARED_WORKSPACE = 4;

// ASCII lookup tables. Code units above 127 are word characters normalized to
// "_" (95), so every normalized code unit, and every neighboring pair of them,
// fits a 128-entry row.
const ASCII_KIND = new Uint8Array(128).fill(TREE_PART_SYMBOL);
const WORD_FLAGS = new Uint8Array(128);
const PAIR_CODES = new Uint8Array(128 * 128);
const SYMBOL_PAIR_HASHES = new Uint8Array(128 * 128);
for (let code = 0; code < 128; code++) {
  if (code === 10 || code === 13) ASCII_KIND[code] = TREE_PART_NEWLINE;
  else if (code === 9 || code === 11 || code === 12 || code === 32) ASCII_KIND[code] = TREE_PART_SPACE;
  else if (code >= 97 && code <= 122) WORD_FLAGS[code] = 1;
  else if (code >= 65 && code <= 90) WORD_FLAGS[code] = 2;
  else if (code >= 48 && code <= 57) WORD_FLAGS[code] = 4;
  else if (code === 95) WORD_FLAGS[code] = 8;
  if (WORD_FLAGS[code]) ASCII_KIND[code] = TREE_PART_WORD;
  for (let right = 0; right < 128; right++) {
    // Zero remains an edge/no-neighbor sentinel; 31 buckets cover every unseen
    // neighboring boundary without a growing hand-authored operator table.
    SYMBOL_PAIR_HASHES[code << 7 | right] = 1 + ((Math.imul(code + 1, 131) ^ right) >>> 0) % 31;
  }
}
// Pair codes 1-14 in order. Code 15 is never emitted.
const PAIRS = "///**/--=>::</{{${}}->???.<>";
for (let index = 0; index < PAIRS.length; index += 2) {
  PAIR_CODES[PAIRS.charCodeAt(index) << 7 | PAIRS.charCodeAt(index + 1)] = index / 2 + 1;
}

function createWorkspace() {
  return [
    INITIAL_CAPACITY,
    new Uint32Array(INITIAL_CAPACITY * TREE_FEATURE_STRIDE),
    new Uint32Array(INITIAL_CAPACITY * 2),
  ];
}

// Grow from the parts-per-code-unit ratio seen so far, so large sources need
// one or two reallocations instead of one per doubling. A source never has
// more parts than code units.
function growWorkspace(workspace, count, offset, length) {
  const estimate = Math.ceil(count / offset * length * 1.0625) + INITIAL_CAPACITY;
  const capacity = Math.max(count + 1, Math.min(length, Math.max(workspace[WORKSPACE_CAPACITY] * 2, estimate)));
  const data = new Uint32Array(capacity * TREE_FEATURE_STRIDE);
  const ranges = new Uint32Array(capacity * 2);
  data.set(workspace[WORKSPACE_DATA].subarray(0, count * TREE_FEATURE_STRIDE));
  ranges.set(workspace[WORKSPACE_RANGES].subarray(0, count * 2));
  workspace[WORKSPACE_CAPACITY] = capacity;
  workspace[WORKSPACE_DATA] = data;
  workspace[WORKSPACE_RANGES] = ranges;
}

/** Split source into words, horizontal-space runs, individual newlines, and symbols. */
export function prepareTreeSource(source) {
  const workspace = pool.pop() ?? createWorkspace();
  const length = source.length;
  let capacity = workspace[WORKSPACE_CAPACITY];
  let data = workspace[WORKSPACE_DATA];
  let ranges = workspace[WORKSPACE_RANGES];
  let count = 0;
  let offset = 0;
  let lineStart = true;
  let previousLast = 0;
  let previousKind = -1;
  while (offset < length) {
    const from = offset;
    const firstRaw = source.charCodeAt(offset++);
    const first = firstRaw > 127 ? 95 : firstRaw;
    const kind = firstRaw > 127 ? TREE_PART_WORD : ASCII_KIND[firstRaw];
    let flags = lineStart ? 16 : 0;
    let last = first;
    let packed = kind | first << 5;
    if (kind === TREE_PART_WORD) {
      let hash = Math.imul(2166136261 ^ first, 16777619);
      let secondaryHash = Math.imul(0x9e3779b9 ^ first, 2246822519);
      flags |= WORD_FLAGS[first];
      while (offset < length) {
        const raw = source.charCodeAt(offset);
        let normalized = 95;
        if (raw <= 127) {
          if (ASCII_KIND[raw] !== TREE_PART_WORD) break;
          normalized = raw;
        }
        hash = Math.imul(hash ^ normalized, 16777619);
        secondaryHash = Math.imul(secondaryHash ^ normalized, 2246822519);
        flags |= WORD_FLAGS[normalized];
        last = normalized;
        offset += 1;
      }
      packed |= (hash & 255) << 19;
      flags |= (secondaryHash & 127) << 15;
    } else if (kind === TREE_PART_SPACE) {
      if (firstRaw === 9) flags |= 32;
      while (offset < length) {
        const raw = source.charCodeAt(offset);
        if (raw > 127 || ASCII_KIND[raw] !== TREE_PART_SPACE) break;
        if (raw === 9) flags |= 32;
        last = raw;
        offset += 1;
      }
    } else if (kind === TREE_PART_NEWLINE) {
      if (firstRaw === 13 && offset < length && source.charCodeAt(offset) === 10) {
        last = 10;
        offset += 1;
      }
      lineStart = true;
    } else if (firstRaw === 92) {
      flags |= 64;
    }
    if (kind === TREE_PART_WORD || kind === TREE_PART_SYMBOL) lineStart = false;

    if (count === capacity) {
      growWorkspace(workspace, count, offset, length);
      capacity = workspace[WORKSPACE_CAPACITY];
      data = workspace[WORKSPACE_DATA];
      ranges = workspace[WORKSPACE_RANGES];
    }
    const base = count * TREE_FEATURE_STRIDE;
    const partLength = offset - from;
    packed |= (partLength > 127 ? 7 : 31 - Math.clz32(partLength)) << 2 | last << 12;
    if (count > 0) {
      const pairIndex = previousLast << 7 | first;
      const pair = PAIR_CODES[pairIndex];
      let previousContext = pair << 11;
      flags |= pair << 7;
      if (kind === TREE_PART_SYMBOL || previousKind === TREE_PART_SYMBOL) {
        const generic = SYMBOL_PAIR_HASHES[pairIndex];
        flags |= generic << 22;
        previousContext |= generic << 27;
      }
      data[base - 1] |= previousContext;
    }
    data[base] = packed;
    data[base + 1] = flags;
    ranges[count * 2] = from;
    ranges[count * 2 + 1] = offset;
    count += 1;
    previousLast = last;
    previousKind = kind;
  }
  // Compact internal tuple: data, ranges, streams, token count, workspace.
  return [
    data.subarray(0, count * TREE_FEATURE_STRIDE),
    ranges.subarray(0, count * 2),
    Uint32Array.of(0, count), count, workspace,
  ];
}

export function releaseTreePrepared(prepared) {
  if (!prepared?.[PREPARED_WORKSPACE]) return;
  if (pool.length < MAX_POOLED_WORKSPACES) pool.push(prepared[PREPARED_WORKSPACE]);
  prepared[PREPARED_WORKSPACE] = null;
}
