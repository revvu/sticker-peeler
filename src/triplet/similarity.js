import { SOLVED_STATE_KEY } from "../cube/CubeState.js";
import { encodeStateKey, loadEncoderWeights } from "../jepa/cubeEncoder.js";

let encoderWeights = null;
let solvedEmbedding = null;
let initPromise = null;

export function cosineSimilarity(left, right) {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }

  const denominator = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  if (denominator === 0) {
    return 0;
  }

  return dot / denominator;
}

export function initTripletEncoder(url = "assets/triplet/encoder_weights.json") {
  if (!initPromise) {
    initPromise = loadEncoderWeights(url)
      .then((weights) => {
        encoderWeights = weights;
        solvedEmbedding = encodeStateKey(SOLVED_STATE_KEY, weights);
      })
      .catch((error) => {
        console.warn("Triplet encoder weights unavailable:", error);
        encoderWeights = null;
        solvedEmbedding = null;
      });
  }

  return initPromise;
}

export function isTripletEncoderReady() {
  return Boolean(encoderWeights && solvedEmbedding);
}

export function getSimilarityToSolved(stateKey) {
  if (!encoderWeights || !solvedEmbedding) {
    return null;
  }

  const currentEmbedding = encodeStateKey(stateKey, encoderWeights);
  return cosineSimilarity(currentEmbedding, solvedEmbedding);
}
