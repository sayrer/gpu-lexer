const copies = 10;
// Each sample is a fresh dedicated worker with one warm-up call, so startup
// state and GC pressure from earlier samples do not carry over. The engine
// order rotates every round and the reported time is the median.
const samples = Number(new URLSearchParams(location.search).get("samples") ?? 3);
const sourceUrl = "https://unpkg.com/three@0.97.0/build/three.min.js";
const allEngines = [
  ["gpu-lexer", "./gpu-lexer.js?run=10"],
  ["Prism.js", "./prism.js?run=10"],
  ["Highlight.js", "./highlight.js?run=10"],
  ["Sugar High", "./sugar-high.js?run=10"],
  ["Starry Night", "./starry-night.js?run=10"],
  ["Shiki", "./shiki.js?run=10"],
];
// ?engines=gpu-lexer,Prism.js runs a subset, e.g. for before/after comparisons.
const selected = new URLSearchParams(location.search).get("engines")?.split(",");
const engines = selected ? allEngines.filter(([name]) => selected.includes(name)) : allEngines;
const status = document.querySelector("#status");
const result = document.querySelector("#result");

run().catch((error) => {
  status.textContent = "failed";
  result.textContent = error?.stack ?? String(error);
  const output = { status: "failed", error: error?.message ?? String(error) };
  globalThis.__benchmarkResult = output;
  void report(output);
});

async function run() {
  const response = await fetch(sourceUrl);
  if (!response.ok) throw new Error(`source download failed: HTTP ${response.status}`);
  const unit = await response.text();
  if (unit.length !== 555_650) throw new Error(`unexpected three.min.js length: ${unit.length}`);
  const source = unit.repeat(copies);
  const sampleTimes = Object.fromEntries(engines.map(([name]) => [name, []]));
  for (let round = 0; round < samples; round++) {
    const order = engines.map((_, index) => engines[(index + round) % engines.length]);
    for (const [name, workerUrl] of order) {
      status.textContent = `${name}: sample ${round + 1}/${samples}…`;
      await report({ status: status.textContent, copies, characters: source.length, samples: sampleTimes });
      sampleTimes[name].push(await runWorker(name, workerUrl, source));
      result.textContent = JSON.stringify(sampleTimes, null, 2);
    }
  }
  const times = Object.fromEntries(Object.entries(sampleTimes).map(([name, values]) => [name, median(values)]));
  const output = {
    copies,
    characters: source.length,
    source: sourceUrl,
    warmups: 1,
    measuredRuns: samples,
    statistic: "median of fresh-worker samples",
    userAgent: navigator.userAgent,
    times,
    samples: sampleTimes,
  };
  status.textContent = "done";
  result.textContent = JSON.stringify(output, null, 2);
  globalThis.__benchmarkResult = output;
  await report({ status: "done", ...output });
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function report(value) {
  return fetch("/result", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(value),
  });
}

function runWorker(name, url, source) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(url, { type: "module" });
    worker.onmessage = ({ data }) => {
      worker.terminate();
      if (data.error) reject(new Error(`${name} worker failed: ${data.error}`));
      else resolve(data.elapsed);
    };
    worker.onerror = ({ message, filename, lineno }) => {
      worker.terminate();
      const location = filename ? ` (${filename}:${lineno})` : "";
      reject(new Error(`${name} worker failed${location}: ${message}`));
    };
    worker.postMessage(source);
  });
}
