const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = __dirname;

function exists(filePath) {
  return fs.existsSync(filePath);
}

function getPythonCommand() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const localVenvPython = path.join(ROOT, "env", "Scripts", "python.exe");
  if (exists(localVenvPython)) return localVenvPython;
  return "python";
}

function findLatestBestWeights() {
  const runsDir = path.join(ROOT, "runs", "detect", "runs_signature");
  if (!exists(runsDir)) return null;

  const candidates = [];
  for (const runName of fs.readdirSync(runsDir)) {
    const bestPath = path.join(runsDir, runName, "weights", "best.pt");
    if (exists(bestPath)) {
      candidates.push({
        file: bestPath,
        mtimeMs: fs.statSync(bestPath).mtimeMs,
      });
    }
  }

  candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return candidates[0]?.file || null;
}

function parseArgs(argv) {
  const args = {
    image: argv[2],
    weights: process.env.WEIGHTS || findLatestBestWeights(),
    conf: process.env.CONF || "0.15",
    imgsz: process.env.IMGSZ || "1024",
    save: process.env.SAVE || path.join(ROOT, "node_prediction_result.jpg"),
  };

  for (let i = 3; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!value) continue;

    if (key === "--weights") args.weights = path.resolve(ROOT, value);
    if (key === "--conf") args.conf = value;
    if (key === "--imgsz") args.imgsz = value;
    if (key === "--save") args.save = path.resolve(ROOT, value);
    i += 1;
  }

  return args;
}

async function main() {
  const args = parseArgs(process.argv);

  if (!args.image) {
    console.error("Cach dung: node predict_with_node.js <duong-dan-anh> [--weights best.pt] [--conf 0.15] [--imgsz 1024] [--save output.jpg]");
    process.exit(1);
  }

  if (!args.weights || !exists(args.weights)) {
    console.error("Khong tim thay file weights. Hay truyen --weights runs/.../weights/best.pt");
    process.exit(1);
  }

  const imagePath = path.resolve(ROOT, args.image);
  if (!exists(imagePath)) {
    console.error(`Khong tim thay anh: ${imagePath}`);
    process.exit(1);
  }

  const python = getPythonCommand();
  const script = path.join(ROOT, "predict_single_image.py");
  const commandArgs = [
    script,
    "--weights",
    args.weights,
    "--image",
    imagePath,
    "--conf",
    args.conf,
    "--imgsz",
    args.imgsz,
    "--save",
    args.save,
  ];

  console.log(`Dang dung weights: ${args.weights}`);
  console.log(`Dang predict anh : ${imagePath}`);

  const child = spawn(python, commandArgs, {
    cwd: ROOT,
    stdio: "inherit",
    env: {
      ...process.env,
      MPLBACKEND: "Agg",
    },
  });

  child.on("error", (error) => {
    console.error(`Khong chay duoc Python: ${python}`);
    console.error(error.message);
    process.exit(1);
  });

  child.on("exit", (code) => {
    if (code === 0) {
      console.log(`\nXong. Anh ket qua: ${args.save}`);
      return;
    }
    process.exit(code || 1);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
