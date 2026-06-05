import { gelu, linear } from "../jepa/cubeEncoder.js";
import { indicesToFaceGrid, stateKeyToFaceGrid } from "./faceGridFeaturize.js";

const DEFAULT_FACE_NEIGHBORS = {
  0: [2, 4, 3, 5],
  1: [2, 4, 3, 5],
  2: [0, 4, 1, 5],
  3: [0, 4, 1, 5],
  4: [2, 1, 3, 0],
  5: [2, 1, 3, 0]
};

function relu(value) {
  return value > 0 ? value : 0;
}

function softplus(value) {
  if (value > 20) {
    return value;
  }
  return Math.log1p(Math.exp(value));
}

function conv2d(input, channels, height, width, layer) {
  const { weight, bias, kernel_size: kernelSize, out_channels: outChannels } = layer;
  const padding = layer.padding ?? Math.floor(kernelSize / 2);
  const output = new Float32Array(outChannels * height * width);

  for (let outChannel = 0; outChannel < outChannels; outChannel += 1) {
    const kernel = weight[outChannel];

    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        let sum = bias[outChannel];

        for (let inChannel = 0; inChannel < channels; inChannel += 1) {
          const channelOffset = inChannel * height * width;

          for (let kernelRow = 0; kernelRow < kernelSize; kernelRow += 1) {
            for (let kernelColumn = 0; kernelColumn < kernelSize; kernelColumn += 1) {
              const sourceRow = row + kernelRow - padding;
              const sourceColumn = column + kernelColumn - padding;

              if (sourceRow < 0 || sourceColumn < 0 || sourceRow >= height || sourceColumn >= width) {
                continue;
              }

              const inputIndex = channelOffset + sourceRow * width + sourceColumn;
              sum += input[inputIndex] * kernel[inChannel][kernelRow][kernelColumn];
            }
          }
        }

        output[outChannel * height * width + row * width + column] = sum;
      }
    }
  }

  return output;
}

function embedSingleFace(faceRows, stickerEmbed) {
  const embedDim = stickerEmbed[0].length;
  const height = 3;
  const width = 3;
  const embedded = new Float32Array(embedDim * height * width);

  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const faceEmbedding = stickerEmbed[faceRows[row][column]];
      const offset = row * width + column;

      for (let dimension = 0; dimension < embedDim; dimension += 1) {
        embedded[dimension * height * width + offset] = faceEmbedding[dimension];
      }
    }
  }

  return embedded;
}

function runConvStack(input, channels, convLayers) {
  let current = input;
  let currentChannels = channels;
  const height = 3;
  const width = 3;

  for (let layerIndex = 0; layerIndex < convLayers.length; layerIndex += 1) {
    const layer = convLayers[layerIndex];
    current = conv2d(current, currentChannels, height, width, layer);
    currentChannels = layer.out_channels;

    for (let index = 0; index < current.length; index += 1) {
      current[index] = gelu(current[index]);
    }
  }

  return current;
}

function getFaceNeighbors(weights) {
  if (!weights.face_neighbors) {
    return DEFAULT_FACE_NEIGHBORS;
  }

  const neighbors = {};
  for (const [faceIndex, values] of Object.entries(weights.face_neighbors)) {
    neighbors[Number(faceIndex)] = values;
  }
  return neighbors;
}

export function encodeFaceGrid(faceGrid, weights) {
  const faceFeatures = [];

  for (let faceIndex = 0; faceIndex < 6; faceIndex += 1) {
    const embedded = embedSingleFace(faceGrid[faceIndex], weights.sticker_embed);
    const convOut = runConvStack(embedded, weights.sticker_embed_dim, weights.face_conv);
    faceFeatures.push(convOut);
  }

  const faceNeighbors = getFaceNeighbors(weights);
  const convChannels = weights.conv_channels;
  const faceFeatDim = convChannels * 9;
  const fusedFaces = [];

  for (let faceIndex = 0; faceIndex < 6; faceIndex += 1) {
    const own = faceFeatures[faceIndex];
    const neighbors = faceNeighbors[faceIndex];
    const neighborMean = new Float32Array(faceFeatDim);

    for (const neighborIndex of neighbors) {
      const neighbor = faceFeatures[neighborIndex];
      for (let index = 0; index < faceFeatDim; index += 1) {
        neighborMean[index] += neighbor[index] / neighbors.length;
      }
    }

    const fusionInput = new Float32Array(faceFeatDim * 2);
    fusionInput.set(own, 0);
    fusionInput.set(neighborMean, faceFeatDim);

    let hidden = fusionInput;
    for (let layerIndex = 0; layerIndex < weights.fusion_mlp.length; layerIndex += 1) {
      const layer = weights.fusion_mlp[layerIndex];
      hidden = linear(hidden, layer.weight, layer.bias);
      if (layerIndex < weights.fusion_mlp.length - 1) {
        for (let index = 0; index < hidden.length; index += 1) {
          hidden[index] = gelu(hidden[index]);
        }
      }
    }

    for (let index = 0; index < hidden.length; index += 1) {
      hidden[index] = gelu(hidden[index]);
    }

    fusedFaces.push(hidden);
  }

  let output = new Float32Array(fusedFaces[0].length * fusedFaces.length);
  let offset = 0;
  for (const fused of fusedFaces) {
    output.set(fused, offset);
    offset += fused.length;
  }

  for (let layerIndex = 0; layerIndex < weights.output_mlp.length; layerIndex += 1) {
    const layer = weights.output_mlp[layerIndex];
    output = linear(output, layer.weight, layer.bias);
    if (layerIndex < weights.output_mlp.length - 1) {
      for (let index = 0; index < output.length; index += 1) {
        output[index] = gelu(output[index]);
      }
    }
  }

  for (let index = 0; index < output.length; index += 1) {
    output[index] = softplus(output[index]);
  }

  return output;
}

export function encodeStateKey(stateKey, weights) {
  return encodeFaceGrid(stateKeyToFaceGrid(stateKey), weights);
}

export function orderDistanceToSolved(embedding, solvedEmbedding) {
  let distance = 0;

  for (let index = 0; index < embedding.length; index += 1) {
    distance += relu(embedding[index] - solvedEmbedding[index]);
  }

  return distance;
}

export async function loadOrderEmbeddingWeights(url = "assets/order-embedding/encoder_weights.json") {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load order-embedding weights (${response.status})`);
  }
  return response.json();
}
