import {
  encodeStateKey,
  loadOrderEmbeddingWeights,
  orderDistanceToSolved
} from "./cnnEncoder.js";

let modelWeights = null;
let initPromise = null;

export function initOrderEmbeddingModel(url = "assets/order-embedding/encoder_weights.json") {
  if (!initPromise) {
    initPromise = loadOrderEmbeddingWeights(url)
      .then((weights) => {
        modelWeights = weights;
      })
      .catch((error) => {
        console.warn("Order-embedding model weights unavailable:", error);
        modelWeights = null;
      });
  }

  return initPromise;
}

export function isOrderEmbeddingModelReady() {
  return Boolean(modelWeights);
}

export function predictOrderDistanceToSolved(stateKey) {
  if (!modelWeights) {
    return null;
  }

  const embedding = encodeStateKey(stateKey, modelWeights);
  const solvedEmbedding = new Float32Array(modelWeights.solved_embedding);
  return orderDistanceToSolved(embedding, solvedEmbedding);
}
