import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  createSolvedCubeState,
  getTurnCommandsForNotation,
  normalizeMove
} from "./cube/CubeState.js";
import { treeSearch } from "./solvers/utils/treeSearch.js";
import { availableSolvers } from "./solvers/index.js";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x151719);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(5, 4.5, 6);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0x2f3540, 2.2));

const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(6, 8, 5);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xc7ddff, 0.9);
fillLight.position.set(-5, 2, -4);
scene.add(fillLight);

const bottomLight = new THREE.DirectionalLight(0xffffff, 1.2);
bottomLight.position.set(-3, -6, 4);
scene.add(bottomLight);

const cubelets = [];
const cubeletSize = 0.96;
const stickerSize = 0.86;
const stickerOffset = cubeletSize / 2 + 0.006;
const turnDuration = 260;
const scrambleTurnDuration = turnDuration / 2;
const quarterTurn = Math.PI / 2;
const reusableVector = new THREE.Vector3();
const reusableMatrix = new THREE.Matrix4();

const stickerColors = {
  right: 0xd8292f,
  left: 0xff8a00,
  up: 0xf8fafc,
  down: 0xffd000,
  front: 0x1fbf5b,
  back: 0x2563eb,
  hidden: 0x101214
};

const moveQueue = [];
const scrambleButton = document.querySelector("#scrambleButton");
const scrambleSequence = document.querySelector("#scrambleSequence");
const solverSelect = document.querySelector("#solverSelect");
const solverStepButton = document.querySelector("#solverStepButton");
const solverStepLog = document.querySelector("#solverStepLog");
const solverConsole = document.querySelector("#solverConsole");
const scrambleFaces = ["F", "B", "U", "D", "R", "L"];
const scrambleSuffixes = ["", "'", "2"];
const moveSoundUrl = "assets/rubik-move.mp3";
const soundSliceDuration = 0.16;
const moveSoundOffsets = [
  1.080, 1.280, 1.880, 2.080, 2.400, 3.480, 4.000, 6.800,
  8.840, 9.600, 9.880, 10.600, 11.080, 11.560, 12.440, 13.600,
  13.840, 14.080, 14.440, 15.240, 16.200, 17.640, 18.600, 19.560
];

let audioContext = null;
let audioMaster = null;
let moveSoundBuffer = null;
let moveSoundLoadPromise = null;
let activeTurn = null;
let logicalCubeState = createSolvedCubeState();
let solverStepInProgress = false;
let solverStepCount = 0;

function createStickerMaterial(color) {
  return new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.08,
    roughness: 0.54,
    metalness: 0.02
  });
}

const bodyMaterial = new THREE.MeshStandardMaterial({
  color: stickerColors.hidden,
  roughness: 0.62,
  metalness: 0.02
});

function addSticker(cubelet, color, position, rotation) {
  const sticker = new THREE.Mesh(
    new THREE.PlaneGeometry(stickerSize, stickerSize),
    createStickerMaterial(color)
  );

  sticker.position.copy(position);
  sticker.rotation.set(rotation.x, rotation.y, rotation.z);
  cubelet.add(sticker);
}

function createCubelet(x, y, z) {
  const cubelet = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(cubeletSize, cubeletSize, cubeletSize), bodyMaterial);
  cubelet.add(body);

  if (x === 1) {
    addSticker(cubelet, stickerColors.right, new THREE.Vector3(stickerOffset, 0, 0), new THREE.Euler(0, Math.PI / 2, 0));
  }

  if (x === -1) {
    addSticker(cubelet, stickerColors.left, new THREE.Vector3(-stickerOffset, 0, 0), new THREE.Euler(0, -Math.PI / 2, 0));
  }

  if (y === 1) {
    addSticker(cubelet, stickerColors.up, new THREE.Vector3(0, stickerOffset, 0), new THREE.Euler(-Math.PI / 2, 0, 0));
  }

  if (y === -1) {
    addSticker(cubelet, stickerColors.down, new THREE.Vector3(0, -stickerOffset, 0), new THREE.Euler(Math.PI / 2, 0, 0));
  }

  if (z === 1) {
    addSticker(cubelet, stickerColors.front, new THREE.Vector3(0, 0, stickerOffset), new THREE.Euler(0, 0, 0));
  }

  if (z === -1) {
    addSticker(cubelet, stickerColors.back, new THREE.Vector3(0, 0, -stickerOffset), new THREE.Euler(0, Math.PI, 0));
  }

  cubelet.position.set(x, y, z);
  cubelet.userData.grid = new THREE.Vector3(x, y, z);
  scene.add(cubelet);
  cubelets.push(cubelet);
}

for (let x = -1; x <= 1; x += 1) {
  for (let y = -1; y <= 1; y += 1) {
    for (let z = -1; z <= 1; z += 1) {
      if (x !== 0 || y !== 0 || z !== 0) {
        createCubelet(x, y, z);
      }
    }
  }
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function getFaceCubelets(axis, layer) {
  return cubelets.filter((cubelet) => cubelet.userData.grid[axis] === layer);
}

function rotateGridPosition(grid, axis, direction) {
  const { x, y, z } = grid;

  if (axis === "x") {
    grid.y = -direction * z;
    grid.z = direction * y;
  } else if (axis === "y") {
    grid.x = direction * z;
    grid.z = -direction * x;
  } else {
    grid.x = -direction * y;
    grid.y = direction * x;
  }
}

function snapNumber(value) {
  return Math.round(value);
}

function snapBasisVector(x, y, z) {
  reusableVector.set(x, y, z);
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);

  if (ax >= ay && ax >= az) {
    return new THREE.Vector3(Math.sign(x) || 1, 0, 0);
  }

  if (ay >= ax && ay >= az) {
    return new THREE.Vector3(0, Math.sign(y) || 1, 0);
  }

  return new THREE.Vector3(0, 0, Math.sign(z) || 1);
}

function snapQuaternionToRightAngles(cubelet) {
  reusableMatrix.makeRotationFromQuaternion(cubelet.quaternion);
  const elements = reusableMatrix.elements;
  const xAxis = snapBasisVector(elements[0], elements[1], elements[2]);
  const yAxis = snapBasisVector(elements[4], elements[5], elements[6]);
  const zAxis = snapBasisVector(elements[8], elements[9], elements[10]);

  reusableMatrix.makeBasis(xAxis, yAxis, zAxis);
  cubelet.quaternion.setFromRotationMatrix(reusableMatrix).normalize();
}

async function unlockAudio() {
  if (!audioContext) {
    audioContext = new AudioContext();
    const compressor = audioContext.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value = 18;
    compressor.ratio.value = 5;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.08;
    compressor.connect(audioContext.destination);
    audioMaster = compressor;
  }

  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  if (!moveSoundLoadPromise) {
    moveSoundLoadPromise = fetch(moveSoundUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load ${moveSoundUrl}`);
        }
        return response.arrayBuffer();
      })
      .then((audioData) => audioContext.decodeAudioData(audioData))
      .then((buffer) => {
        moveSoundBuffer = buffer;
      })
      .catch((error) => {
        console.warn(error);
      });
  }
}

function playMoveSound(duration = turnDuration) {
  void unlockAudio();

  if (!audioContext || !moveSoundBuffer) {
    return;
  }

  const source = audioContext.createBufferSource();
  const gain = audioContext.createGain();
  const now = audioContext.currentTime;
  const sliceDuration = Math.min(soundSliceDuration, duration / 1000);
  const offsets = moveSoundOffsets.filter((offset) => offset + sliceDuration < moveSoundBuffer.duration);
  const offset = offsets[Math.floor(Math.random() * offsets.length)] ?? 0;

  source.buffer = moveSoundBuffer;
  source.playbackRate.value = 0.96 + Math.random() * 0.1;
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(1.8, now + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + sliceDuration);
  source.connect(gain).connect(audioMaster);
  source.start(now, offset, sliceDuration);
}

function isTurnBusy() {
  return Boolean(activeTurn || moveQueue.length > 0);
}

function getSelectedSolver() {
  return availableSolvers.find((solver) => solver.id === solverSelect.value) ?? availableSolvers[0];
}

function updateControls() {
  const busy = isTurnBusy() || solverStepInProgress;
  scrambleButton.disabled = busy;
  solverSelect.disabled = busy;
  solverStepButton.disabled = busy || logicalCubeState.isSolved();
  solverStepButton.title = logicalCubeState.isSolved() ? "Cube solved" : "";
}

function clearSolverLog() {
  solverStepCount = 0;
  solverStepLog.replaceChildren();
}

function appendSolverLog(text) {
  solverStepCount += 1;
  const item = document.createElement("li");
  item.textContent = text;
  solverStepLog.appendChild(item);
  solverStepLog.scrollTop = solverStepLog.scrollHeight;
}

function getElementBlockSize(element) {
  const style = getComputedStyle(element);
  const marginTop = parseFloat(style.marginTop) || 0;
  const marginBottom = parseFloat(style.marginBottom) || 0;

  return element.offsetHeight + marginTop + marginBottom;
}

function updateConsoleOverflow() {
  const contentHeight = Array.from(solverConsole.children).reduce(
    (total, child) => total + getElementBlockSize(child),
    0
  );

  solverConsole.classList.toggle("is-overflowing", contentHeight > solverConsole.clientHeight);
  solverConsole.scrollTop = solverConsole.scrollHeight;
}

function appendConsoleMessage(text) {
  const item = document.createElement("li");
  const time = document.createElement("time");
  const message = document.createElement("span");
  const now = new Date();

  item.className = "console-message";
  time.dateTime = now.toISOString();
  time.textContent = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
  message.textContent = text;
  item.append(time, message);
  solverConsole.appendChild(item);
  updateConsoleOverflow();
}

function attachSolverConsole() {
  for (const solver of availableSolvers) {
    solver.setConsoleSink((message) => appendConsoleMessage(message));
  }
}

function finalizeTurn(turn) {
  turn.pivot.rotation[turn.axis] = turn.angle;
  turn.pivot.updateMatrixWorld(true);

  turn.face.forEach((cubelet) => {
    scene.attach(cubelet);
    rotateGridPosition(cubelet.userData.grid, turn.axis, turn.direction);
    cubelet.position.set(
      snapNumber(cubelet.userData.grid.x),
      snapNumber(cubelet.userData.grid.y),
      snapNumber(cubelet.userData.grid.z)
    );
    snapQuaternionToRightAngles(cubelet);
    cubelet.updateMatrixWorld(true);
  });

  scene.remove(turn.pivot);
  activeTurn = null;
  updateControls();
  startNextTurn();
}

function startNextTurn() {
  if (activeTurn || moveQueue.length === 0) {
    updateControls();
    return;
  }

  const move = moveQueue.shift();
  const face = getFaceCubelets(move.axis, move.layer);
  const pivot = new THREE.Object3D();
  const angle = move.direction * quarterTurn;

  scene.add(pivot);
  pivot.updateMatrixWorld(true);
  face.forEach((cubelet) => pivot.attach(cubelet));

  activeTurn = {
    ...move,
    angle,
    face,
    pivot,
    duration: move.duration ?? turnDuration,
    startedAt: performance.now()
  };
  playMoveSound(activeTurn.duration);
  updateControls();
}

function enqueueTurn(axis, layer, direction, duration = turnDuration) {
  moveQueue.push({ axis, layer, direction, duration });
  updateControls();
  startNextTurn();
}

function runNotationMoves(moves, duration = turnDuration) {
  const normalizedMoves = moves.map(normalizeMove);

  for (const move of normalizedMoves) {
    logicalCubeState.applyMove(move);
    const turnCommands = getTurnCommandsForNotation(move, duration);

    for (const command of turnCommands) {
      enqueueTurn(command.axis, command.layer, command.direction, command.duration);
    }
  }

  updateControls();
}

function generateScramble(length = 20) {
  const moves = [];
  let previousFace = "";

  while (moves.length < length) {
    const face = scrambleFaces[Math.floor(Math.random() * scrambleFaces.length)];
    if (face === previousFace) {
      continue;
    }

    const suffix = scrambleSuffixes[Math.floor(Math.random() * scrambleSuffixes.length)];
    moves.push(`${face}${suffix}`);
    previousFace = face;
  }

  return moves;
}

function runScramble() {
  if (isTurnBusy()) {
    return;
  }

  const scramble = generateScramble();
  clearSolverLog();
  scrambleSequence.textContent = scramble.join(" ");
  runNotationMoves(scramble, scrambleTurnDuration);
}

async function runSolverStep() {
  if (isTurnBusy() || solverStepInProgress || logicalCubeState.isSolved()) {
    return;
  }

  const solver = getSelectedSolver();
  solverStepInProgress = true;
  updateControls();

  try {
    const moves = await Promise.resolve(solver.step(logicalCubeState.clone()));

    if (!Array.isArray(moves) || moves.length === 0) {
      appendSolverLog(solver.noResultLabel);
      return;
    }

    const normalizedMoves = moves.map(normalizeMove);
    appendSolverLog(normalizedMoves.join(" "));
    runNotationMoves(normalizedMoves, turnDuration);
  } catch (error) {
    console.error(error);
    appendSolverLog("solver error");
  } finally {
    solverStepInProgress = false;
    updateControls();
  }
}

function populateSolvers() {
  solverSelect.replaceChildren();

  for (const solver of availableSolvers) {
    const option = document.createElement("option");
    option.value = solver.id;
    option.textContent = solver.label;
    solverSelect.appendChild(option);
  }
}

window.addEventListener("keydown", (event) => {
  const face = event.key.toUpperCase();
  if (!scrambleFaces.includes(face) || event.metaKey || event.ctrlKey || event.altKey) {
    return;
  }

  event.preventDefault();
  clearSolverLog();
  const notation = `${face}${event.shiftKey ? "'" : ""}`;
  runNotationMoves([notation], turnDuration);
});

scrambleButton.addEventListener("click", runScramble);
solverStepButton.addEventListener("click", runSolverStep);

function animate(now) {
  requestAnimationFrame(animate);

  if (activeTurn) {
    const elapsed = now - activeTurn.startedAt;
    const t = Math.min(elapsed / activeTurn.duration, 1);
    activeTurn.pivot.rotation[activeTurn.axis] = activeTurn.angle * easeInOutCubic(t);

    if (t >= 1) {
      finalizeTurn(activeTurn);
    }
  }

  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  updateConsoleOverflow();
});

attachSolverConsole();
populateSolvers();
updateControls();

window.cubeDebug = {
  cubelets,
  enqueueTurn,
  generateScramble,
  runScramble,
  runNotationMoves,
  treeSearch,
  availableSolvers,
  appendConsoleMessage,
  moveQueue,
  get logicalCubeState() {
    return logicalCubeState.clone();
  },
  get activeTurn() {
    return activeTurn;
  }
};

requestAnimationFrame(animate);
