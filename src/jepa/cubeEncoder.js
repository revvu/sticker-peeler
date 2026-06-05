import { stateKeyToIndices } from "./stateFeaturize.js";

export function gelu(value) {
  const cube = value * value * value;
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * cube)));
}

export function linear(input, weight, bias) {
  const outputSize = weight.length;
  const output = new Float32Array(outputSize);

  for (let row = 0; row < outputSize; row += 1) {
    let sum = bias[row];
    const rowWeights = weight[row];
    for (let column = 0; column < input.length; column += 1) {
      sum += input[column] * rowWeights[column];
    }
    output[row] = sum;
  }

  return output;
}

function embedStickerIndices(indices, stickerEmbed) {
  const stickerCount = indices.length;
  const embedDim = stickerEmbed[0].length;
  const flattened = new Float32Array(stickerCount * embedDim);

  for (let stickerIndex = 0; stickerIndex < stickerCount; stickerIndex += 1) {
    const faceEmbedding = stickerEmbed[indices[stickerIndex]];
    const offset = stickerIndex * embedDim;
    for (let dimension = 0; dimension < embedDim; dimension += 1) {
      flattened[offset + dimension] = faceEmbedding[dimension];
    }
  }

  return flattened;
}

export function encodeIndices(indices, weights) {
  let hidden = embedStickerIndices(indices, weights.sticker_embed);

  for (let layerIndex = 0; layerIndex < weights.mlp.length; layerIndex += 1) {
    const layer = weights.mlp[layerIndex];
    hidden = linear(hidden, layer.weight, layer.bias);
    if (layerIndex < weights.mlp.length - 1) {
      for (let index = 0; index < hidden.length; index += 1) {
        hidden[index] = gelu(hidden[index]);
      }
    }
  }

  return hidden;
}

export function encodeStateKey(stateKey, weights) {
  return encodeIndices(stateKeyToIndices(stateKey), weights);
}

export async function loadEncoderWeights(url = "assets/jepa/encoder_weights.json") {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load encoder weights (${response.status})`);
  }
  return response.json();
}
