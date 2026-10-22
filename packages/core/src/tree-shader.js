import { runtimeTensorLayout, validTreeContext } from "./model-layout.js";
import {
  TREE_SECONDARY_HASH_BUCKETS, TREE_SYMBOL_PAIR_BUCKETS,
  isTreeFeatureVersion, runtimeTreeFeatureLayout, treeInputSize, treeScaleBuckets,
} from "./tree-features.js";

const BLOCK = 32;
const HIDDEN = 32;
const HEAP = BLOCK * 2 - 1;
// Share the block's workgroup storage across four tokens at a time. Each
// token keeps its original channel arithmetic and ordered recurrent updates.
const LOCAL_TOKEN_LANES = 4;
// The classifier runs on the same 128-thread workgroup as the block tree:
// sixteen styled parts at a time, eight lanes each.
const CLASSIFIER_TOKENS = 16;
const CLASSIFIER_LANES = 8;
// Lanes of the neighbor prefix scan; one workgroup walks a stream's blocks in
// chunks of this many.
const NEIGHBOR_LANES = 256;
const SENTINEL = "0xffffffffu";

export function createTreeShader(model, { f16 = false } = {}) {
  const hidden = model.hiddenSize;
  const classifier = model.architecture.classifierDimensions;
  const auxiliary = model.architecture.auxiliaryStates.length;
  const outputs = model.outputSize;
  const hashBuckets = model.architecture.lexemeHashBuckets ?? 128;
  const featureVersion = model.featureVersion;
  const scaleBuckets = model.architecture.scaleBuckets;
  const featureLayout = runtimeTreeFeatureLayout(hashBuckets, featureVersion);
  if (!validTreeContext(model) || model.model !== "hierarchical-tree" ||
      !isTreeFeatureVersion(featureVersion) || model.inputSize !== treeInputSize(hashBuckets, featureVersion) ||
      hidden !== HIDDEN || classifier < 1 || classifier > 256 ||
      model.architecture.tree !== "scale-aware-butterfly-binary" ||
      model.architecture.blockParts !== BLOCK || scaleBuckets !== treeScaleBuckets(featureVersion) ||
      (featureVersion >= 3 &&
        (model.architecture.secondaryLexemeHashBuckets !== TREE_SECONDARY_HASH_BUCKETS ||
         model.architecture.neighborSymbolHashBuckets !== TREE_SYMBOL_PAIR_BUCKETS))) {
    throw new Error("promoted hierarchical-tree model is incompatible with this runtime");
  }
  const tensors = Object.fromEntries(runtimeTensorLayout(model).map((value) => [value.name, value]));
  const offset = (name) => {
    if (!tensors[name]) throw new Error(`promoted hierarchical-tree model is missing ${name}`);
    return tensors[name].offset;
  };
  const contextType = f16 ? "f16" : "f32";
  const cast = (value) => f16 ? `f16(${value})` : value;
  const read = (value) => f16 ? `f32(${value})` : value;
  const quantize = (value) => f16 ? `f32(f16(${value}))` : value;
  const enable = f16 ? "enable f16;\n" : "";
  const partnerDepth = Math.log2(hidden) - 1;
  const hashFeature = hashBuckets ? `
  if (kind == 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.primaryHash}u + (((packed >> 19u) & 255u) & ${hashBuckets - 1}u)) * ${hidden}u + h];${featureVersion >= 3 ? `
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.secondaryHash}u + ((context >> 15u) & 127u)) * ${hidden}u + h];` : ""}
  }` : "";
  const symbolPairFeatures = featureVersion >= 3 ? `
  let previous_symbol_pair = (context >> 22u) & 31u;
  let next_symbol_pair = (context >> 27u) & 31u;
  if (previous_symbol_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.previousSymbolPair}u + previous_symbol_pair - 1u) * ${hidden}u + h];
  }
  if (next_symbol_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.nextSymbolPair}u + next_symbol_pair - 1u) * ${hidden}u + h];
  }` : "";

  return /* wgsl */ `${enable}
struct Params { stream_count: u32, token_count: u32, block_count: u32, tree_nodes: u32, }
struct TreeStream { start: u32, count: u32, block_start: u32, block_count: u32, tree_offset: u32, tree_power: u32, }
struct TreeBlock { start: u32, count: u32, stream: u32, local_index: u32, }
@group(0) @binding(0) var<storage, read> features: array<u32>;
@group(0) @binding(1) var<storage, read_write> labels: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> streams: array<TreeStream>;
@group(0) @binding(4) var<storage, read> weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> leaf_states: array<${contextType}>;
@group(0) @binding(6) var<storage, read_write> tree_up: array<${contextType}>;
// Read-only view of hybrid_neighbors for the classifier. Loads through a
// read-only binding at a uniform index count as uniform, which lets the
// per-block styled count bound a loop that contains barriers.
@group(0) @binding(7) var<storage, read> block_masks: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> reserved_8: array<u32>;
@group(0) @binding(9) var<storage, read> blocks: array<TreeBlock>;
@group(0) @binding(10) var<storage, read_write> hybrid_local: array<f32>;
@group(0) @binding(11) var<storage, read_write> scratch: array<f32>;
// Per block: x = first styled part after the block, y = last styled part
// before it (both ${SENTINEL} when none), z = bitmask of the block's own
// styled parts. Styled means a word or symbol: not a space run or newline.
@group(0) @binding(12) var<storage, read_write> hybrid_neighbors: array<vec4<u32>>;
var<workgroup> hybrid_work: array<vec4<f32>, ${BLOCK * hidden}>;
var<workgroup> neighbor_work: array<u32, ${NEIGHBOR_LANES}>;
var<workgroup> local_tree: array<f32, ${HEAP * hidden}>;
var<workgroup> compact: array<u32, ${BLOCK}>;
var<workgroup> classifier_auxiliary: array<f32, ${CLASSIFIER_TOKENS * auxiliary}>;
var<workgroup> classifier_projected: array<f32, ${CLASSIFIER_TOKENS * classifier}>;
var<workgroup> classifier_scores: array<f32, ${CLASSIFIER_TOKENS * outputs}>;

fn flat_group(id: vec3<u32>) -> u32 { return id.x + id.y * 65535u; }
fn flat_id(id: vec3<u32>) -> u32 { return id.x + id.y * 4194240u; }
fn sigmoid(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
fn raw_state(token: u32, start: u32, end: u32, h: u32) -> f32 {
  if (token < start || token >= end) { return 0.0; }
  let packed = features[token * 2u];
  let context = features[token * 2u + 1u];
  let kind = packed & 3u;
  var sum = weights[${offset("featureEmbedding")}u + kind * ${hidden}u + h];
  sum += weights[${offset("featureEmbedding")}u + (4u + ((packed >> 2u) & 7u)) * ${hidden}u + h];
  sum += weights[${offset("featureEmbedding")}u + (12u + ((packed >> 5u) & 127u)) * ${hidden}u + h];
  sum += weights[${offset("featureEmbedding")}u + (140u + ((packed >> 12u) & 127u)) * ${hidden}u + h];${hashFeature}
  let flag_offset = ${featureLayout.flags}u;
  let flags = context & 127u;
  for (var bit = 0u; bit < 7u; bit++) {
    if ((flags & (1u << bit)) != 0u) {
      sum += weights[${offset("featureEmbedding")}u + (flag_offset + bit) * ${hidden}u + h];
    }
  }
  let previous_pair = (context >> 7u) & 15u;
  let next_pair = (context >> 11u) & 15u;
  if (previous_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.previousPair}u + previous_pair - 1u) * ${hidden}u + h];
  }
  if (next_pair != 0u) {
    sum += weights[${offset("featureEmbedding")}u + (${featureLayout.nextPair}u + next_pair - 1u) * ${hidden}u + h];
  }${symbolPairFeatures}
  return sum;
}

fn local_state(token: u32, h: u32, start: u32, end: u32, neighbors: vec2<f32>) -> f32 {
  var sum = weights[${offset("leafBias")}u + h];
  for (var d = 0u; d < 5u; d++) {
    if (token + d >= start + 2u && token + d < end + 2u) {
      sum += raw_state(token + d - 2u, start, end, h) *
        weights[${offset("localOffsetScale")}u + d * ${hidden}u + h];
    }
  }
  sum += raw_state(bitcast<u32>(neighbors.x), start, end, h) * weights[${offset("localNonspaceScale")}u + h];
  sum += raw_state(bitcast<u32>(neighbors.y), start, end, h) * weights[${offset("localNonspaceScale") + hidden}u + h];
  return tanh(sum);
}

fn affine_at(part: u32, h: u32) -> vec2<f32> {
  var input = weights[${offset("stateInputBias")}u + h];
  var gate = weights[${offset("stateGateBias")}u + h];
  for (var c = 0u; c < ${hidden}u; c++) {
    let value = hybrid_work[part * ${hidden}u + c].x;
    input += value * weights[${offset("stateInput")}u + h * ${hidden}u + c];
    gate += value * weights[${offset("stateGate")}u + h * ${hidden}u + c];
  }
  let a = sigmoid(gate);
  return vec2<f32>(a, (1.0 - a) * tanh(input));
}

fn merge_state(left: f32, right: f32, left_cross: f32, right_cross: f32, h: u32, depth: u32) -> f32 {
  let scale = min(depth, ${scaleBuckets - 1}u) * ${hidden}u + h;
  let mixed = tanh(
    left * weights[${offset("mergeOwnLeft")}u + scale] +
    right * weights[${offset("mergeOwnRight")}u + scale] +
    left_cross * weights[${offset("mergeCrossLeft")}u + scale] +
    right_cross * weights[${offset("mergeCrossRight")}u + scale] +
    weights[${offset("mergeBias")}u + scale]
  );
  let boundary = select(right, left, h < ${hidden / 2}u);
  return (mixed + boundary) * 0.5;
}

fn descend_state(
  parent: f32, own: f32, sibling: f32,
  parent_cross: f32, own_cross: f32, sibling_cross: f32,
  h: u32, depth: u32, right: bool,
) -> f32 {
  let scale = min(depth, ${scaleBuckets - 1}u) * ${hidden}u + h;
  let bias = select(weights[${offset("downLeftBias")}u + scale], weights[${offset("downRightBias")}u + scale], right);
  let mixed = tanh(
    parent * weights[${offset("downOwnParent")}u + scale] +
    own * weights[${offset("downOwnSelf")}u + scale] +
    sibling * weights[${offset("downOwnSibling")}u + scale] +
    parent_cross * weights[${offset("downCrossParent")}u + scale] +
    own_cross * weights[${offset("downCrossSelf")}u + scale] +
    sibling_cross * weights[${offset("downCrossSibling")}u + scale] + bias
  );
  let skip = sigmoid(weights[${offset("downSkip")}u + scale]);
  return parent * skip + mixed * (1.0 - skip);
}

// One bit per part of a block, set for words and symbols. Neighboring lanes
// hold different blocks, so the loop runs a fixed ${BLOCK} steps and masks the
// tail block with select instead of a shorter, divergent trip count.
@compute @workgroup_size(64)
fn hybrid_neighbor_blocks(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = flat_id(id);
  if (index >= params.block_count) { return; }
  let block = blocks[index];
  var mask = 0u;
  for (var i = 0u; i < ${BLOCK}u; i++) {
    let token = block.start + min(i, block.count - 1u);
    let kind = features[token * 2u] & 3u;
    let styled = i < block.count && kind != 1u && kind != 2u;
    mask |= u32(styled) << i;
  }
  let first = select(${SENTINEL}, block.start + countTrailingZeros(mask), mask != 0u);
  let last = select(${SENTINEL}, block.start + ${BLOCK - 1}u - countLeadingZeros(mask), mask != 0u);
  hybrid_neighbors[index] = vec4<u32>(first, last, mask, 0u);
}

// Exclusive max scan across the workgroup's ${NEIGHBOR_LANES} lanes. Returns
// the exclusive value for this lane and the total over every lane.
fn neighbor_scan_max(lane: u32, input: u32) -> vec2<u32> {
  var value = input;
  neighbor_work[lane] = value;
  for (var step = 1u; step < ${NEIGHBOR_LANES}u; step = step << 1u) {
    workgroupBarrier();
    var previous = 0u;
    if (lane >= step) { previous = neighbor_work[lane - step]; }
    workgroupBarrier();
    value = max(value, previous);
    neighbor_work[lane] = value;
  }
  workgroupBarrier();
  var before = 0u;
  if (lane > 0u) { before = neighbor_work[lane - 1u]; }
  let total = neighbor_work[${NEIGHBOR_LANES - 1}u];
  workgroupBarrier();
  return vec2<u32>(before, total);
}

// The min counterpart, with ${SENTINEL} as the identity.
fn neighbor_scan_min(lane: u32, input: u32) -> vec2<u32> {
  var value = input;
  neighbor_work[lane] = value;
  for (var step = 1u; step < ${NEIGHBOR_LANES}u; step = step << 1u) {
    workgroupBarrier();
    var previous = ${SENTINEL};
    if (lane >= step) { previous = neighbor_work[lane - step]; }
    workgroupBarrier();
    value = min(value, previous);
    neighbor_work[lane] = value;
  }
  workgroupBarrier();
  var before = ${SENTINEL};
  if (lane > 0u) { before = neighbor_work[lane - 1u]; }
  let total = neighbor_work[${NEIGHBOR_LANES - 1}u];
  workgroupBarrier();
  return vec2<u32>(before, total);
}

// Turns each block's own first and last styled part into the nearest styled
// parts outside the block: an exclusive prefix max of (last + 1, zero for
// none) and an exclusive suffix min of first (${SENTINEL} for none). One
// workgroup scans a stream ${NEIGHBOR_LANES} blocks at a time and carries the
// running extreme between chunks. The two directions run as separate passes
// and each lane rewrites its block's whole entry, so no two lanes ever store
// into the same vector at once.
@compute @workgroup_size(${NEIGHBOR_LANES})
fn hybrid_neighbor_prefixes(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let stream_index = flat_group(group);
  if (stream_index >= params.stream_count) { return; }
  let stream = streams[stream_index];
  let lane = local.x;
  var previous = 0u;
  for (var chunk = 0u; chunk < stream.block_count; chunk += ${NEIGHBOR_LANES}u) {
    let index = chunk + lane;
    let valid = index < stream.block_count;
    var entry = vec4<u32>(0u);
    var last = 0u;
    if (valid) {
      entry = hybrid_neighbors[stream.block_start + index];
      last = select(0u, entry.y + 1u, entry.y != ${SENTINEL});
    }
    let scan = neighbor_scan_max(lane, last);
    if (valid) {
      hybrid_neighbors[stream.block_start + index] =
        vec4<u32>(entry.x, max(scan.x, previous) - 1u, entry.z, entry.w);
    }
    previous = max(scan.y, previous);
  }
  var next = ${SENTINEL};
  for (var chunk = 0u; chunk < stream.block_count; chunk += ${NEIGHBOR_LANES}u) {
    let position = chunk + lane;
    let valid = position < stream.block_count;
    let index = stream.block_count - 1u - min(position, stream.block_count - 1u);
    var entry = vec4<u32>(0u);
    var first = ${SENTINEL};
    if (valid) {
      entry = hybrid_neighbors[stream.block_start + index];
      first = entry.x;
    }
    let scan = neighbor_scan_min(lane, first);
    if (valid) {
      hybrid_neighbors[stream.block_start + index] =
        vec4<u32>(min(scan.x, next), entry.y, entry.z, entry.w);
    }
    next = min(scan.y, next);
  }
}

@compute @workgroup_size(${hidden * LOCAL_TOKEN_LANES})
fn hybrid_scan_blocks(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let index = flat_group(group);
  if (index >= params.block_count) { return; }
  let block = blocks[index];
  let stream = streams[block.stream];
  let h = local.x & ${hidden - 1}u;
  let token_lane = local.x >> 5u;
  if (local.x < block.count) {
    // Nearest styled parts from the block mask: the highest set bit below
    // this part and the lowest set bit above it, falling back to the
    // stream-level neighbors when the block has none on that side.
    let neighbors = hybrid_neighbors[index];
    let below = neighbors.z & ((1u << local.x) - 1u);
    let above = neighbors.z & ~((2u << local.x) - 1u);
    let previous = select(neighbors.y, block.start + ${BLOCK - 1}u - countLeadingZeros(below), below != 0u);
    let next = select(neighbors.x, block.start + countTrailingZeros(above), above != 0u);
    hybrid_work[local.x * ${hidden}u].y = bitcast<f32>(previous);
    hybrid_work[local.x * ${hidden}u].z = bitcast<f32>(next);
  }
  workgroupBarrier();
  for (var i = token_lane; i < block.count; i += ${LOCAL_TOKEN_LANES}u) {
    let value = local_state(block.start + i, h, stream.start, stream.start + stream.count,
      hybrid_work[i * ${hidden}u].yz);
    hybrid_local[(block.start + i) * ${hidden}u + h] = value;
    hybrid_work[i * ${hidden}u + h].x = value;
  }
  workgroupBarrier();
  for (var i = token_lane; i < block.count; i += ${LOCAL_TOKEN_LANES}u) {
    let pair = affine_at(i, h);
    hybrid_work[i * ${hidden}u + h].y = pair.x;
    hybrid_work[i * ${hidden}u + h].z = pair.y;
  }
  workgroupBarrier();
  if (token_lane == 0u) {
    var forward = vec2<f32>(1.0, 0.0);
    for (var i = 0u; i < block.count; i++) {
      let value = hybrid_work[i * ${hidden}u + h];
      forward = vec2<f32>(value.y * forward.x, value.y * forward.y + value.z);
    }
    var reverse = vec2<f32>(1.0, 0.0);
    for (var i = block.count; i > 0u; i--) {
      let value = hybrid_work[(i - 1u) * ${hidden}u + h];
      reverse = vec2<f32>(value.y * reverse.x, value.y * reverse.y + value.z);
    }
    let at = (index * ${hidden}u + h) * 4u;
    scratch[at] = forward.x;
    scratch[at + 1u] = forward.y;
    scratch[at + 2u] = reverse.x;
    scratch[at + 3u] = reverse.y;
  }
}

@compute @workgroup_size(${hidden * 8})
fn hybrid_scan_prefixes(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let index = flat_group(group);
  let stream_index = index >> 2u;
  if (stream_index >= params.stream_count) { return; }
  let stream = streams[stream_index];
  let lane = local.x & 31u;
  let h = (index & 3u) * 8u + (local.x >> 5u);
  var forward_carry = 0.0;
  var reverse_carry = 0.0;
  for (var chunk = 0u; chunk < stream.block_count; chunk += 32u) {
    let forward_index = chunk + lane;
    let reverse_index = stream.block_count - 1u - forward_index;
    let valid = forward_index < stream.block_count;
    let forward_at = ((stream.block_start + forward_index) * ${hidden}u + h) * 4u;
    let reverse_at = ((stream.block_start + reverse_index) * ${hidden}u + h) * 4u;
    var value = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    if (valid) {
      value = vec4<f32>(scratch[forward_at], scratch[forward_at + 1u],
        scratch[reverse_at + 2u], scratch[reverse_at + 3u]);
    }
    hybrid_work[local.x] = value;
    for (var step = 1u; step < 32u; step = step << 1u) {
      workgroupBarrier();
      var previous = vec4<f32>(1.0, 0.0, 1.0, 0.0);
      if (lane >= step) { previous = hybrid_work[local.x - step]; }
      workgroupBarrier();
      value = vec4<f32>(value.x * previous.x, value.x * previous.y + value.y,
        value.z * previous.z, value.z * previous.w + value.w);
      hybrid_work[local.x] = value;
    }
    workgroupBarrier();
    var before = vec4<f32>(1.0, 0.0, 1.0, 0.0);
    if (lane > 0u) { before = hybrid_work[local.x - 1u]; }
    if (valid) {
      scratch[forward_at + 1u] = before.x * forward_carry + before.y;
      scratch[reverse_at + 3u] = before.z * reverse_carry + before.w;
    }
    let total = hybrid_work[local.x | 31u];
    forward_carry = total.x * forward_carry + total.y;
    reverse_carry = total.z * reverse_carry + total.w;
    workgroupBarrier();
  }
}

@compute @workgroup_size(${hidden * LOCAL_TOKEN_LANES})
fn hybrid_mix_tree_up(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let index = flat_group(group);
  if (index >= params.block_count) { return; }
  let block = blocks[index];
  let stream = streams[block.stream];
  let h = local.x & ${hidden - 1}u;
  let token_lane = local.x >> 5u;
  for (var i = token_lane; i < block.count; i += ${LOCAL_TOKEN_LANES}u) {
    hybrid_work[i * ${hidden}u + h].x = hybrid_local[(block.start + i) * ${hidden}u + h];
  }
  workgroupBarrier();
  for (var i = token_lane; i < block.count; i += ${LOCAL_TOKEN_LANES}u) {
    let pair = affine_at(i, h);
    hybrid_work[i * ${hidden}u + h].y = pair.x;
    hybrid_work[i * ${hidden}u + h].z = pair.y;
  }
  workgroupBarrier();
  if (token_lane == 0u) {
    let summary = (index * ${hidden}u + h) * 4u;
    var state = scratch[summary + 1u];
    for (var i = 0u; i < block.count; i++) {
      let at = i * ${hidden}u + h;
      let value = hybrid_work[at];
      state = value.y * state + value.z;
      hybrid_work[at].w = state;
    }
    state = scratch[summary + 3u];
    for (var i = block.count; i > 0u; i--) {
      let at = (i - 1u) * ${hidden}u + h;
      let value = hybrid_work[at];
      state = value.y * state + value.z;
      hybrid_work[at].z = state;
    }
  }
  workgroupBarrier();
  for (var i = token_lane; i < block.count; i += ${LOCAL_TOKEN_LANES}u) {
    let token = block.start + i;
    var sum = hybrid_local[token * ${hidden}u + h] + weights[${offset("stateMixBias")}u + h];
    let row = ${offset("stateMix")}u + h * ${hidden * 2}u;
    for (var c = 0u; c < ${hidden}u; c++) {
      let context = hybrid_work[i * ${hidden}u + c];
      sum += context.w * weights[row + c] + context.z * weights[row + ${hidden}u + c];
    }
    let leaf = ${quantize("tanh(sum)")};
    hybrid_work[i * ${hidden}u + h].y = leaf;
    leaf_states[token * ${hidden}u + h] = ${cast("leaf")};
  }
  workgroupBarrier();
  var width = ${BLOCK}u;
  var valid = block.count;
  var depth = 0u;
  loop {
    if (width <= 1u) { break; }
    let parent_width = width / 2u;
    for (var parent = token_lane; parent < parent_width; parent += ${LOCAL_TOKEN_LANES}u) {
      let child = parent * 2u;
      var merged = 0.0;
      if (child < valid) {
        let left_at = child * ${hidden}u;
        let input_y = (depth & 1u) == 0u;
        let left = select(hybrid_work[left_at + h].x, hybrid_work[left_at + h].y, input_y);
        merged = left;
        if (child + 1u < valid) {
          let right_at = (child + 1u) * ${hidden}u;
          let partner = h ^ (1u << depth);
          merged = merge_state(left,
            select(hybrid_work[right_at + h].x, hybrid_work[right_at + h].y, input_y),
            select(hybrid_work[left_at + partner].x, hybrid_work[left_at + partner].y, input_y),
            select(hybrid_work[right_at + partner].x, hybrid_work[right_at + partner].y, input_y), h, depth);
        }
      }
      if ((depth & 1u) == 0u) { hybrid_work[parent * ${hidden}u + h].x = merged; }
      else { hybrid_work[parent * ${hidden}u + h].y = merged; }
    }
    workgroupBarrier();
    width = parent_width;
    valid = (valid + 1u) / 2u;
    depth += 1u;
  }
  let node = stream.tree_offset + stream.tree_power - 1u + block.local_index;
  if (token_lane == 0u) {
    tree_up[node * ${hidden}u + h] = ${cast("hybrid_work[h].x")};
  }
}

@compute @workgroup_size(${hidden * 8})
fn tree_global(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let stream_index = flat_group(group);
  let h = local.x & ${hidden - 1}u;
  let node_lane = local.x >> 5u;
  if (stream_index >= params.stream_count) { return; }
  let stream = streams[stream_index];
  let leaf_base = stream.tree_offset + stream.tree_power - 1u;
  for (var index = stream.block_count + node_lane; index < stream.tree_power; index += 8u) {
    tree_up[(leaf_base + index) * ${hidden}u + h] = ${cast("0.0")};
  }
  storageBarrier(); workgroupBarrier();
  var width = stream.tree_power;
  var valid = stream.block_count;
  var depth = 5u;
  loop {
    if (width <= 1u) { break; }
    let parent_width = width / 2u;
    let parent_base = stream.tree_offset + parent_width - 1u;
    let child_base = stream.tree_offset + width - 1u;
    let partner = h ^ (1u << min(depth, ${partnerDepth}u));
    for (var index = node_lane; index < parent_width; index += 8u) {
      let child = index * 2u;
      var value = 0.0;
      if (child < valid) {
        let left = ${read(`tree_up[(child_base + child) * ${hidden}u + h]`)};
        value = left;
        if (child + 1u < valid) {
          value = merge_state(left, ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + h]`)},
            ${read(`tree_up[(child_base + child) * ${hidden}u + partner]`)},
            ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + partner]`)}, h, depth);
        }
      }
      tree_up[(parent_base + index) * ${hidden}u + h] = ${cast("value")};
    }
    storageBarrier(); workgroupBarrier();
    width = parent_width;
    valid = (valid + 1u) / 2u;
    depth += 1u;
  }
  if (node_lane == 0u) {
    scratch[stream.tree_offset * ${hidden}u + h] = ${quantize(read(`tree_up[stream.tree_offset * ${hidden}u + h]`))};
  }
  storageBarrier(); workgroupBarrier();
  width = 1u;
  var down_depth = 4u + (31u - countLeadingZeros(stream.tree_power));
  loop {
    if (width >= stream.tree_power) { break; }
    let parent_base = stream.tree_offset + width - 1u;
    let child_base = stream.tree_offset + width * 2u - 1u;
    let child_span = stream.tree_power / (width * 2u);
    let child_valid = (stream.block_count + child_span - 1u) / child_span;
    for (var index = node_lane; index < width; index += 8u) {
      let child = index * 2u;
      if (child >= child_valid) { continue; }
      let parent = scratch[(parent_base + index) * ${hidden}u + h];
      let left = ${read(`tree_up[(child_base + child) * ${hidden}u + h]`)};
      if (child + 1u < child_valid) {
        let right = ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + h]`)};
        let partner = h ^ (1u << min(down_depth, ${partnerDepth}u));
        let parent_cross = scratch[(parent_base + index) * ${hidden}u + partner];
        let left_cross = ${read(`tree_up[(child_base + child) * ${hidden}u + partner]`)};
        let right_cross = ${read(`tree_up[(child_base + child + 1u) * ${hidden}u + partner]`)};
        scratch[(child_base + child) * ${hidden}u + h] = ${quantize("descend_state(parent, left, right, parent_cross, left_cross, right_cross, h, down_depth, false)")};
        scratch[(child_base + child + 1u) * ${hidden}u + h] = ${quantize("descend_state(parent, right, left, parent_cross, right_cross, left_cross, h, down_depth, true)")};
      } else {
        scratch[(child_base + child) * ${hidden}u + h] = ${quantize("parent")};
      }
    }
    storageBarrier(); workgroupBarrier();
    width *= 2u;
    down_depth = select(0u, down_depth - 1u, down_depth > 0u);
  }
}

// Rebuilds the block's butterfly tree from its leaves, descends the global
// context through it, then classifies the block's styled parts. The tree
// phases spread parents over four token lanes exactly as hybrid_mix_tree_up
// does, keeping each node's arithmetic unchanged.
@compute @workgroup_size(${hidden * LOCAL_TOKEN_LANES})
fn tree_down_classify(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let block_index = flat_group(group);
  if (block_index >= params.block_count) { return; }
  let block = blocks[block_index];
  let stream = streams[block.stream];
  let h = local.x & ${hidden - 1}u;
  let token_lane = local.x >> 5u;
  for (var part = token_lane; part < block.count; part += ${LOCAL_TOKEN_LANES}u) {
    local_tree[(${BLOCK - 1}u + part) * ${hidden}u + h] =
      ${read(`leaf_states[(block.start + part) * ${hidden}u + h]`)};
  }
  // Styled parts in order, so every classifier slot in a batch has work
  // instead of idling on whitespace.
  let mask = block_masks[block_index].z;
  if (token_lane == 0u && ((mask >> h) & 1u) == 1u) {
    compact[countOneBits(mask & ((1u << h) - 1u))] = h;
  }
  workgroupBarrier();
  var width = ${BLOCK}u;
  var valid = block.count;
  var depth = 0u;
  loop {
    if (width <= 1u) { break; }
    let parent_width = width / 2u;
    let parent_base = parent_width - 1u;
    let child_base = width - 1u;
    for (var parent = token_lane; parent < parent_width; parent += ${LOCAL_TOKEN_LANES}u) {
      let child = parent * 2u;
      var merged = 0.0;
      if (child < valid) {
        let left = local_tree[(child_base + child) * ${hidden}u + h];
        merged = left;
        if (child + 1u < valid) {
          let partner = h ^ (1u << depth);
          merged = merge_state(left, local_tree[(child_base + child + 1u) * ${hidden}u + h],
            local_tree[(child_base + child) * ${hidden}u + partner],
            local_tree[(child_base + child + 1u) * ${hidden}u + partner], h, depth);
        }
      }
      local_tree[(parent_base + parent) * ${hidden}u + h] = merged;
    }
    workgroupBarrier();
    width = parent_width;
    valid = (valid + 1u) / 2u;
    depth += 1u;
  }
  if (token_lane == 0u) {
    let node = stream.tree_offset + stream.tree_power - 1u + block.local_index;
    local_tree[h] = scratch[node * ${hidden}u + h];
  }
  workgroupBarrier();
  width = 1u;
  depth = 4u;
  loop {
    if (width >= ${BLOCK}u) { break; }
    let parent_base = width - 1u;
    let child_base = width * 2u - 1u;
    let child_span = ${BLOCK}u / (width * 2u);
    let child_valid = (block.count + child_span - 1u) / child_span;
    // Descended values wait in registers until every lane has read its
    // partner channel, so overwriting a child never races a neighbor's read.
    var pending: array<f32, ${BLOCK / LOCAL_TOKEN_LANES}>;
    var slot = 0u;
    for (var parent_index = token_lane; parent_index < width; parent_index += ${LOCAL_TOKEN_LANES}u) {
      let child = parent_index * 2u;
      var left_value = 0.0;
      var right_value = 0.0;
      if (child < child_valid) {
        let parent = local_tree[(parent_base + parent_index) * ${hidden}u + h];
        let left = local_tree[(child_base + child) * ${hidden}u + h];
        left_value = parent;
        if (child + 1u < child_valid) {
          let right = local_tree[(child_base + child + 1u) * ${hidden}u + h];
          let partner = h ^ (1u << depth);
          let parent_cross = local_tree[(parent_base + parent_index) * ${hidden}u + partner];
          let left_cross = local_tree[(child_base + child) * ${hidden}u + partner];
          let right_cross = local_tree[(child_base + child + 1u) * ${hidden}u + partner];
          left_value = descend_state(
            parent, left, right, parent_cross, left_cross, right_cross, h, depth, false);
          right_value = descend_state(
            parent, right, left, parent_cross, right_cross, left_cross, h, depth, true);
        }
      }
      pending[slot] = left_value;
      pending[slot + 1u] = right_value;
      slot += 2u;
    }
    workgroupBarrier();
    slot = 0u;
    for (var parent_index = token_lane; parent_index < width; parent_index += ${LOCAL_TOKEN_LANES}u) {
      let child = parent_index * 2u;
      if (child < child_valid) {
        local_tree[(child_base + child) * ${hidden}u + h] = pending[slot];
        if (child + 1u < child_valid) {
          local_tree[(child_base + child + 1u) * ${hidden}u + h] = pending[slot + 1u];
        }
      }
      slot += 2u;
    }
    workgroupBarrier();
    width *= 2u;
    depth = select(0u, depth - 1u, depth > 0u);
  }

  let styled_count = countOneBits(mask);
  let slot = local.x / ${CLASSIFIER_LANES}u;
  let lane = local.x % ${CLASSIFIER_LANES}u;
  for (var batch = 0u; batch < ${BLOCK / CLASSIFIER_TOKENS}u; batch++) {
    // styled_count is uniform across the workgroup, so this exit is too.
    if (batch * ${CLASSIFIER_TOKENS}u >= styled_count) { break; }
    let rank = batch * ${CLASSIFIER_TOKENS}u + slot;
    let token_valid = rank < styled_count;
    let part = compact[min(rank, styled_count - 1u)];
    let token = block.start + part;
    for (var a = lane; token_valid && a < ${auxiliary}u; a += ${CLASSIFIER_LANES}u) {
      var value = weights[${offset("auxiliaryBias")}u + a];
      for (var feature_channel = 0u; feature_channel < ${hidden}u; feature_channel++) {
        let leaf = ${read(`leaf_states[token * ${hidden}u + feature_channel]`)};
        let context = ${quantize(`local_tree[(${BLOCK - 1}u + part) * ${hidden}u + feature_channel]`)};
        let row = ${offset("auxiliaryOutput")}u + a * ${hidden * 2}u;
        value += weights[row + feature_channel] * leaf + weights[row + ${hidden}u + feature_channel] * context;
      }
      classifier_auxiliary[slot * ${auxiliary}u + a] = sigmoid(value);
    }
    workgroupBarrier();
    for (var c = lane; token_valid && c < ${classifier}u; c += ${CLASSIFIER_LANES}u) {
      var value = weights[${offset("classifierBias")}u + c];
      for (var feature_channel = 0u; feature_channel < ${hidden}u; feature_channel++) {
        let leaf = ${read(`leaf_states[token * ${hidden}u + feature_channel]`)};
        let context = ${quantize(`local_tree[(${BLOCK - 1}u + part) * ${hidden}u + feature_channel]`)};
        let row = ${offset("classifierInput")}u + c * ${hidden * 2 + auxiliary}u;
        value += weights[row + feature_channel] * leaf + weights[row + ${hidden}u + feature_channel] * context;
      }
      let columns = ${offset("classifierInput")}u + c * ${hidden * 2 + auxiliary}u + ${hidden * 2}u;
      for (var a = 0u; a < ${auxiliary}u; a++) {
        value += weights[columns + a] * classifier_auxiliary[slot * ${auxiliary}u + a];
      }
      classifier_projected[slot * ${classifier}u + c] = tanh(value);
    }
    workgroupBarrier();
    for (var output = lane; token_valid && output < ${outputs}u; output += ${CLASSIFIER_LANES}u) {
      var score = weights[${offset("outputBias")}u + output];
      for (var c = 0u; c < ${classifier}u; c++) {
        score += weights[${offset("output")}u + output * ${classifier}u + c] *
          classifier_projected[slot * ${classifier}u + c];
      }
      classifier_scores[slot * ${outputs}u + output] = score;
    }
    workgroupBarrier();
    // Whitespace parts keep the cleared label (plain) without a write.
    if (lane == 0u && token_valid) {
      var selected = 0u;
      var selected_score = classifier_scores[slot * ${outputs}u];
      for (var output = 1u; output < ${outputs}u; output++) {
        let score = classifier_scores[slot * ${outputs}u + output];
        if (score > selected_score) { selected_score = score; selected = output; }
      }
      let ignored = atomicOr(&labels[token / 4u], selected << ((token & 3u) * 8u));
    }
    workgroupBarrier();
  }
}
`;
}
