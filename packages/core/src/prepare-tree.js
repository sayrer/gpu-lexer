import {
  TREE_FEATURE_STRIDE, TREE_PART_NEWLINE, TREE_PART_SPACE, TREE_PART_SYMBOL, TREE_PART_WORD,
} from "./constants.js";

const INITIAL_CAPACITY = 512;
const MAX_POOLED_WORKSPACES = 4;
const pool = [];
const WORKSPACE_CAPACITY = 0;
const WORKSPACE_DATA = 1;
const WORKSPACE_RANGES = 2;
const WORKSPACE_BYTES = 3;
const PREPARED_WORKSPACE = 4;

// Character class of each ASCII code; codes above 127 are word characters.
const CLASS = new Uint8Array(128).fill(TREE_PART_SYMBOL);
for (let code = 48; code <= 57; code++) CLASS[code] = TREE_PART_WORD;
for (let code = 65; code <= 90; code++) CLASS[code] = TREE_PART_WORD;
for (let code = 97; code <= 122; code++) CLASS[code] = TREE_PART_WORD;
CLASS[95] = TREE_PART_WORD;
for (const code of [9, 11, 12, 32]) CLASS[code] = TREE_PART_SPACE;
CLASS[10] = TREE_PART_NEWLINE;
CLASS[13] = TREE_PART_NEWLINE;

// Shape flag contributed by each normalized word character.
const WORD_FLAG = new Uint8Array(128);
for (let code = 97; code <= 122; code++) WORD_FLAG[code] = 1;
for (let code = 65; code <= 90; code++) WORD_FLAG[code] = 2;
for (let code = 48; code <= 57; code++) WORD_FLAG[code] = 4;
WORD_FLAG[95] = 8;

// Pair code and generic symbol-pair hash for a boundary between two
// normalized characters, indexed previous * 128 + next.
const PAIR = new Uint8Array(128 * 128);
for (const [left, right, code] of [
  [47, 47, 1], [47, 42, 2], [42, 47, 3], [45, 45, 4], [61, 62, 5], [58, 58, 6], [60, 47, 7],
  [123, 123, 8], [36, 123, 9], [125, 125, 10], [45, 62, 11], [63, 63, 12], [63, 46, 13], [60, 62, 14],
]) {
  PAIR[left * 128 + right] = code;
}
const SYMBOL_PAIR = new Uint8Array(128 * 128);
for (let left = 0; left < 128; left++) {
  for (let right = 0; right < 128; right++) {
    // Zero remains an edge/no-neighbor sentinel; 31 buckets cover every
    // unseen neighboring boundary without a hand-authored operator table.
    SYMBOL_PAIR[left * 128 + right] = 1 + ((Math.imul(left + 1, 131) ^ right) >>> 0) % 31;
  }
}

const encoder = new TextEncoder();

function createWorkspace() {
  return [
    INITIAL_CAPACITY,
    new Uint32Array(INITIAL_CAPACITY * TREE_FEATURE_STRIDE),
    new Uint32Array(INITIAL_CAPACITY * 2),
    new Uint8Array(0),
  ];
}

function growWorkspace(workspace) {
  const capacity = workspace[WORKSPACE_CAPACITY] * 2;
  const data = new Uint32Array(capacity * TREE_FEATURE_STRIDE);
  const ranges = new Uint32Array(capacity * 2);
  data.set(workspace[WORKSPACE_DATA]);
  ranges.set(workspace[WORKSPACE_RANGES]);
  workspace[WORKSPACE_CAPACITY] = capacity;
  workspace[WORKSPACE_DATA] = data;
  workspace[WORKSPACE_RANGES] = ranges;
}

// Scanner state carried between the byte and string scanners:
// [offset, count, lineStart, previousLast, previousKind].
const STATE_OFFSET = 0;
const STATE_COUNT = 1;
const STATE_LINE_START = 2;
const STATE_PREVIOUS_LAST = 3;
const STATE_PREVIOUS_KIND = 4;

/**
 * Split source into words, horizontal-space runs, individual newlines, and
 * symbols. The source is scanned as UTF-8 bytes up to its first non-ASCII
 * character, which avoids the string representation checks behind every
 * charCodeAt, and the string scanner takes over from there. Both scanners
 * produce identical features.
 */
export function prepareTreeSource(source) {
  const workspace = pool.pop() ?? createWorkspace();
  const length = source.length;
  if (workspace[WORKSPACE_BYTES].length < length) {
    workspace[WORKSPACE_BYTES] = new Uint8Array(Math.max(length, workspace[WORKSPACE_BYTES].length * 2));
  }
  const bytes = workspace[WORKSPACE_BYTES];
  const { written } = encoder.encodeInto(source, bytes);
  const state = [0, 0, 1, 0, -1];
  scanBytes(bytes, written, workspace, state);
  if (state[STATE_OFFSET] < length) scanString(source, workspace, state);
  const count = state[STATE_COUNT];
  // Compact internal tuple: data, ranges, streams, token count, workspace.
  return [
    workspace[WORKSPACE_DATA].subarray(0, count * TREE_FEATURE_STRIDE),
    workspace[WORKSPACE_RANGES].subarray(0, count * 2),
    Uint32Array.of(0, count), count, workspace,
  ];
}

// Records one part. `first` and `last` are normalized characters.
function emit(workspace, count, kind, from, to, first, last, flags, hash, secondaryHash, previousLast, previousKind) {
  if (count === workspace[WORKSPACE_CAPACITY]) growWorkspace(workspace);
  const data = workspace[WORKSPACE_DATA];
  const ranges = workspace[WORKSPACE_RANGES];
  const base = count * TREE_FEATURE_STRIDE;
  const length = to - from;
  const bucket = length >= 128 ? 7 : 31 - Math.clz32(length);
  let context = flags | (kind === TREE_PART_WORD ? (secondaryHash & 127) << 15 : 0);
  if (count > 0) {
    const boundary = previousLast * 128 + first;
    const pair = PAIR[boundary];
    if (pair !== 0) {
      context |= pair << 7;
      data[base - 1] |= pair << 11;
    }
    if (kind === TREE_PART_SYMBOL || previousKind === TREE_PART_SYMBOL) {
      const generic = SYMBOL_PAIR[boundary];
      context |= generic << 22;
      data[base - 1] |= generic << 27;
    }
  }
  data[base] = kind | (bucket << 2) | (first << 5) | (last << 12) | ((hash & 255) << 19);
  data[base + 1] = context;
  ranges[count * 2] = from;
  ranges[count * 2 + 1] = to;
}

// Scans ASCII bytes until the end or the first byte of a non-ASCII character,
// leaving the state at the start of the part that contains it.
function scanBytes(bytes, length, workspace, state) {
  let count = state[STATE_COUNT];
  let offset = state[STATE_OFFSET];
  let lineStart = state[STATE_LINE_START] === 1;
  let previousLast = state[STATE_PREVIOUS_LAST];
  let previousKind = state[STATE_PREVIOUS_KIND];
  let code = offset < length ? bytes[offset] : 0;
  while (offset < length) {
    const from = offset;
    const first = code;
    if (first > 127) break;
    const kind = CLASS[first];
    let flags = lineStart ? 16 : 0;
    let last = first;
    let hash = 0;
    let secondaryHash = 0;
    if (kind === TREE_PART_WORD) {
      hash = 2166136261;
      secondaryHash = 0x9e3779b9;
      for (;;) {
        last = code;
        hash = Math.imul(hash ^ code, 16777619);
        secondaryHash = Math.imul(secondaryHash ^ code, 2246822519);
        flags |= WORD_FLAG[code];
        offset += 1;
        if (offset >= length) { code = 0; break; }
        code = bytes[offset];
        if (code > 127 || CLASS[code] !== TREE_PART_WORD) break;
      }
      // A word that runs into a non-ASCII character belongs to the string
      // scanner, from its start.
      if (code > 127) { offset = from; break; }
    } else if (kind === TREE_PART_SPACE) {
      for (;;) {
        last = code;
        if (code === 9) flags |= 32;
        offset += 1;
        if (offset >= length) { code = 0; break; }
        code = bytes[offset];
        if (code > 127 || CLASS[code] !== TREE_PART_SPACE) break;
      }
    } else if (kind === TREE_PART_NEWLINE) {
      offset += 1;
      code = offset < length ? bytes[offset] : 0;
      if (first === 13 && code === 10) {
        last = 10;
        offset += 1;
        code = offset < length ? bytes[offset] : 0;
      }
    } else {
      if (first === 92) flags |= 64;
      offset += 1;
      code = offset < length ? bytes[offset] : 0;
    }
    emit(workspace, count, kind, from, offset, first, last, flags, hash, secondaryHash, previousLast, previousKind);
    count += 1;
    previousLast = last;
    previousKind = kind;
    if (kind === TREE_PART_NEWLINE) lineStart = true;
    else if (kind !== TREE_PART_SPACE) lineStart = false;
  }
  state[STATE_OFFSET] = offset;
  state[STATE_COUNT] = count;
  state[STATE_LINE_START] = lineStart ? 1 : 0;
  state[STATE_PREVIOUS_LAST] = previousLast;
  state[STATE_PREVIOUS_KIND] = previousKind;
}

// Scans UTF-16 code units from the state's offset to the end.
function scanString(source, workspace, state) {
  const length = source.length;
  let count = state[STATE_COUNT];
  let offset = state[STATE_OFFSET];
  let lineStart = state[STATE_LINE_START] === 1;
  let previousLast = state[STATE_PREVIOUS_LAST];
  let previousKind = state[STATE_PREVIOUS_KIND];
  let code = offset < length ? source.charCodeAt(offset) : 0;
  while (offset < length) {
    const from = offset;
    const firstRaw = code;
    const first = firstRaw > 127 ? 95 : firstRaw;
    const kind = firstRaw > 127 ? TREE_PART_WORD : CLASS[firstRaw];
    let flags = lineStart ? 16 : 0;
    let last = first;
    let hash = 0;
    let secondaryHash = 0;
    if (kind === TREE_PART_WORD) {
      hash = 2166136261;
      secondaryHash = 0x9e3779b9;
      let normalized = first;
      for (;;) {
        last = normalized;
        hash = Math.imul(hash ^ normalized, 16777619);
        secondaryHash = Math.imul(secondaryHash ^ normalized, 2246822519);
        flags |= WORD_FLAG[normalized];
        offset += 1;
        if (offset >= length) { code = 0; break; }
        code = source.charCodeAt(offset);
        if (code > 127) { normalized = 95; continue; }
        if (CLASS[code] !== TREE_PART_WORD) break;
        normalized = code;
      }
    } else if (kind === TREE_PART_SPACE) {
      for (;;) {
        last = code;
        if (code === 9) flags |= 32;
        offset += 1;
        if (offset >= length) { code = 0; break; }
        code = source.charCodeAt(offset);
        if (code > 127 || CLASS[code] !== TREE_PART_SPACE) break;
      }
    } else if (kind === TREE_PART_NEWLINE) {
      offset += 1;
      code = offset < length ? source.charCodeAt(offset) : 0;
      if (firstRaw === 13 && code === 10) {
        last = 10;
        offset += 1;
        code = offset < length ? source.charCodeAt(offset) : 0;
      }
    } else {
      if (firstRaw === 92) flags |= 64;
      offset += 1;
      code = offset < length ? source.charCodeAt(offset) : 0;
    }
    emit(workspace, count, kind, from, offset, first, last, flags, hash, secondaryHash, previousLast, previousKind);
    count += 1;
    previousLast = last;
    previousKind = kind;
    if (kind === TREE_PART_NEWLINE) lineStart = true;
    else if (kind !== TREE_PART_SPACE) lineStart = false;
  }
  state[STATE_OFFSET] = offset;
  state[STATE_COUNT] = count;
}

export function releaseTreePrepared(prepared) {
  if (!prepared?.[PREPARED_WORKSPACE]) return;
  if (pool.length < MAX_POOLED_WORKSPACES) pool.push(prepared[PREPARED_WORKSPACE]);
  prepared[PREPARED_WORKSPACE] = null;
}
