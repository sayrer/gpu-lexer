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
    const transpose = transposedShape(model, name);
    if (transpose && transpose[0] * transpose[1] !== length) {
      throw new Error(`invalid runtime matrix shape for ${name}`);
    }
    const tensor = { name, offset, length, scale: source.scale, sourceRanges, transpose };
    offset += length;
    return tensor;
  });
}

// The shader reads these [rows][columns] matrices with the row (output unit)
// varying across adjacent threads. The runtime buffer stores them column-major
// so those threads read adjacent weights; values and summation order are
// unchanged.
function transposedShape(model, name) {
  const hidden = model.hiddenSize;
  const classifier = model.architecture.classifierDimensions;
  const auxiliary = model.architecture.auxiliaryStates.length;
  if (name === "stateInput" || name === "stateGate") return [hidden, hidden];
  if (name === "stateMix") return [hidden, hidden * 2];
  if (name === "classifierInput") return [classifier, hidden * 2 + auxiliary];
  if (name === "auxiliaryOutput") return [auxiliary, hidden * 2];
  if (name === "output") return [model.outputSize, classifier];
  return undefined;
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
