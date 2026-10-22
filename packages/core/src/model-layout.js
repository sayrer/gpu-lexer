import {
  runtimeTreeFeatureLayout, unreachableTreeFeatureRows,
} from "./tree-features.js";

// Runtime order is independent from the training checkpoint schema. It was
// selected against the promoted model's Brotli-11 output and keeps subsequent
// offsets deterministic across shader and weight generation.
const RUNTIME_TENSOR_ORDER = [
  "featureEmbedding", "leafBias", "mergeOwnLeft", "mergeOwnRight",
  "mergeCrossLeft", "mergeCrossRight", "mergeBias", "output",
  "downOwnSelf", "downOwnSibling", "downCrossParent", "downCrossSelf",
  "downCrossSibling", "downSkip", "downLeftBias", "downRightBias",
  "classifierInput", "classifierBias", "downOwnParent", "outputBias",
  "auxiliaryOutput", "auxiliaryBias", "localOffsetScale", "localNonspaceScale",
  "stateInput", "stateInputBias", "stateGate", "stateGateBias", "stateMix",
  "stateMixBias",
];

export function isHybridTree(model) {
  return model?.model === "hierarchical-tree" && model.formatVersion === 9;
}

export function validTreeContext(model) {
  return isHybridTree(model) && model.architecture?.context === "local-affine-tree" &&
    model.architecture?.localRadius === 2;
}

export function validateHybridTensors(model) {
  if (!isHybridTree(model)) return;
  const h = model.hiddenSize;
  const lengths = {
    neighborScale: 3 * h, localOffsetScale: 5 * h, localNonspaceScale: 2 * h,
    stateInput: h * h, stateInputBias: h, stateGate: h * h, stateGateBias: h,
    stateMix: h * 2 * h, stateMixBias: h,
  };
  const tensors = model.quantization?.tensors ?? [];
  for (const [name, length] of Object.entries(lengths)) {
    const matches = tensors.filter((tensor) => tensor.name === name);
    const tensor = matches[0];
    if (matches.length !== 1 || tensor.length !== length || !Number.isInteger(tensor.offset) ||
        tensor.offset < 0 || tensor.offset + length > model.quantization.parameterCount) {
      throw new Error(`invalid hybrid runtime tensor ${name}`);
    }
  }
}

export function runtimeTensorLayout(model) {
  validateHybridTensors(model);
  const sourceTensors = model.quantization.tensors;
  const tensors = new Map(sourceTensors.map((tensor) => [tensor.name, tensor]));
  const expected = new Set([...RUNTIME_TENSOR_ORDER, "neighborScale"]);
  if (sourceTensors.some(({ name }) => !expected.has(name)) ||
      expected.size !== sourceTensors.length) {
    throw new Error("promoted hierarchical-tree tensor layout is incompatible with this runtime");
  }

  const hidden = model.hiddenSize;
  const hashBuckets = model.architecture.lexemeHashBuckets ?? 128;
  const compactFeatures = runtimeTreeFeatureLayout(hashBuckets, model.featureVersion);
  const omittedRows = unreachableTreeFeatureRows(hashBuckets, model.featureVersion);
  let offset = 0;
  return RUNTIME_TENSOR_ORDER.map((name) => {
    const source = tensors.get(name);
    if (!source) throw new Error(`promoted hierarchical-tree model is missing ${name}`);
    const sourceRanges = name === "featureEmbedding"
      ? embeddingSourceRanges(source, hidden, model.inputSize, omittedRows)
      : [source.offset, source.offset + source.length];
    const length = name === "featureEmbedding"
      ? compactFeatures.inputSize * hidden
      : source.length;
    const tensor = { name, offset, length, scale: source.scale, sourceRanges };
    offset += length;
    return tensor;
  });
}

// Copies of some tensors appended after the compact weights, in layouts the
// kernels read more efficiently; values are identical to the originals.
// Column-major copies of the scan matrices put a SIMD group's 32 channels on
// consecutive addresses. The classifier matrix gets a 16-byte-aligned copy
// with a padded row stride when the original cannot be read as vec4 rows.
export function appendedTensorLayout(model) {
  const h = model.hiddenSize;
  const auxiliary = model.architecture.auxiliaryStates.length;
  const classifier = model.architecture.classifierDimensions;
  const layout = runtimeTensorLayout(model);
  const tensors = new Map(layout.map((tensor) => [tensor.name, tensor]));
  const last = layout[layout.length - 1];
  let offset = last.offset + last.length;
  const copies = [
    ["stateInput", h, h, true], ["stateGate", h, h, true], ["stateMix", h, 2 * h, true],
  ];
  const classifierInput = tensors.get("classifierInput");
  const classifierStride = 2 * h + auxiliary;
  if (classifierInput.offset % 4 !== 0 || classifierStride % 4 !== 0) {
    copies.push(["classifierInput", classifier, classifierStride, false]);
  }
  const entries = {};
  for (const [name, rows, cols, transposed] of copies) {
    const source = tensors.get(name);
    if (source.length !== rows * cols) throw new Error(`runtime tensor ${name} has an unexpected shape`);
    offset = (offset + 3) & ~3;
    const stride = transposed ? rows : (cols + 3) & ~3;
    entries[name] = { source: source.offset, rows, cols, stride, transposed, offset };
    offset += transposed ? rows * cols : rows * stride;
  }
  return { count: (offset + 3) & ~3, entries };
}

function embeddingSourceRanges(tensor, hidden, inputSize, omittedRows) {
  if (tensor.length !== inputSize * hidden) {
    throw new Error("invalid feature embedding tensor length");
  }
  const ranges = [];
  let firstRow = 0;
  for (const row of omittedRows) {
    if (row > firstRow) ranges.push(
      tensor.offset + firstRow * hidden,
      tensor.offset + row * hidden,
    );
    firstRow = row + 1;
  }
  if (firstRow < inputSize) ranges.push(
    tensor.offset + firstRow * hidden,
    tensor.offset + inputSize * hidden,
  );
  return ranges;
}
