import { encodeStateKey, loadEncoderWeights } from "../jepa/cubeEncoder.js";

let modelWeights = null;
let initPromise = null;

export function softplus(value) {
  if (value > 20) {
    return value;
  }
  return Math.log1p(Math.exp(value));
}

function predictDistanceFromEmbedding(embedding, weights) {
  const head = weights.distance_head;
  let sum = head.bias[0];

  for (let index = 0; index < embedding.length; index += 1) {
    sum += head.weight[0][index] * embedding[index];
  }

  return softplus(sum);
}

export function initDistanceModel(url = "assets/supervised-distance/encoder_weights.json") {
  if (!initPromise) {
    initPromise = loadEncoderWeights(url)
      .then((weights) => {
        modelWeights = weights;
      })
      .catch((error) => {
        console.warn("Supervised distance model weights unavailable:", error);
        modelWeights = null;
      });
  }

  return initPromise;
}

export function isDistanceModelReady() {
  return Boolean(modelWeights);
}

export function predictDistanceToSolved(stateKey) {
  if (!modelWeights) {
    return null;
  }

  const embedding = encodeStateKey(stateKey, modelWeights);
  return predictDistanceFromEmbedding(embedding, modelWeights);
}
