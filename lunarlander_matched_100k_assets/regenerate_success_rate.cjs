#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { createCanvas } = require("@napi-rs/canvas");

const bundleDir = __dirname;
const sourceCsv = path.join(bundleDir, "matched_100k_all_metrics.csv");
const outputPng = path.join(bundleDir, "04_success_rate_comparison.png");

const modelOrder = ["DQN", "PPO", "PPO-tiny", "QRL", "Quantum DQN"];
const modelColors = {
  DQN: "#d95f02",
  PPO: "#4169e1",
  "PPO-tiny": "#00a676",
  QRL: "#7b3294",
  "Quantum DQN": "#b35806",
};
const modelMarkers = {
  DQN: "circle",
  PPO: "square",
  "PPO-tiny": "triangle",
  QRL: "diamond",
  "Quantum DQN": "cross",
};
const modelOffsets = {
  DQN: -1400,
  PPO: -700,
  "PPO-tiny": 0,
  QRL: 700,
  "Quantum DQN": 1400,
};
const binSize = 10000;
const binCount = 10;
const finalTimestep = 100000;

function parseCsv(text) {
  const records = [];
  let record = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      record.push(field);
      field = "";
    } else if (char === "\n") {
      record.push(field.replace(/\r$/, ""));
      if (record.some((value) => value !== "")) records.push(record);
      record = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field !== "" || record.length > 0) {
    record.push(field.replace(/\r$/, ""));
    records.push(record);
  }
  const headers = records.shift();
  return records.map((values) => Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ""])));
}

function loadAndBin() {
  const rows = parseCsv(fs.readFileSync(sourceCsv, "utf8"));
  const bins = new Map(modelOrder.map((model) => [
    model,
    Array.from({ length: binCount }, () => ({ successes: 0, episodes: 0, seeds: new Set() })),
  ]));
  const finalEvaluations = [];

  for (const row of rows) {
    if (row.included_in_plots !== "yes") continue;
    const model = row.agent_label;
    if (!bins.has(model)) continue;

    if (row.metric_type === "evaluation") {
      finalEvaluations.push({ model, seed: Number(row.seed), success: Number(row.success_rate) });
      continue;
    }
    if (row.metric_type !== "train_episode") continue;

    const timestep = Number(row.timestep);
    const success = Number(row.success_rate);
    const seed = Number(row.seed);
    if (![0, 1, 2].includes(seed)) throw new Error(`Unexpected seed for ${model}: ${seed}`);
    if (success !== 0 && success !== 1) throw new Error(`Expected binary success for ${model} seed ${seed}, found ${success}`);
    if (!Number.isFinite(timestep) || timestep < 0 || timestep > finalTimestep) continue;

    const binIndex = Math.min(Math.floor(timestep / binSize), binCount - 1);
    const bin = bins.get(model)[binIndex];
    bin.successes += success;
    bin.episodes += 1;
    bin.seeds.add(seed);
  }

  let totalSeries = 0;
  for (const model of modelOrder) {
    const modelBins = bins.get(model);
    if (modelBins.length !== 10) throw new Error(`Expected 10 bins for ${model}`);
    const allSeeds = new Set(modelBins.flatMap((bin) => [...bin.seeds]));
    if ([...allSeeds].sort().join(",") !== "0,1,2") throw new Error(`Expected seeds 0,1,2 for ${model}`);
    for (const [index, bin] of modelBins.entries()) {
      if (bin.episodes === 0) {
        bin.rate = Number.NaN;
      } else {
        bin.rate = bin.successes / bin.episodes;
      }
      bin.center = index * binSize + binSize / 2;
    }
    totalSeries += 1;
  }

  if (totalSeries !== 5) throw new Error(`Expected 5 model series, found ${totalSeries}`);
  if (finalEvaluations.length !== 15) throw new Error(`Expected 15 final evaluations, found ${finalEvaluations.length}`);
  if (finalEvaluations.some(({ success }) => success !== 0)) {
    throw new Error("Expected all 15 final evaluation success values to remain zero");
  }

  return { bins, finalEvaluations };
}

function drawMarker(ctx, type, x, y, radius, color) {
  ctx.beginPath();
  if (type === "circle") {
    ctx.arc(x, y, radius, 0, Math.PI * 2);
  } else if (type === "square") {
    ctx.rect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (type === "triangle") {
    ctx.moveTo(x, y - radius * 1.15);
    ctx.lineTo(x + radius * 1.05, y + radius * 0.9);
    ctx.lineTo(x - radius * 1.05, y + radius * 0.9);
    ctx.closePath();
  } else if (type === "diamond") {
    ctx.moveTo(x, y - radius * 1.2);
    ctx.lineTo(x + radius, y);
    ctx.lineTo(x, y + radius * 1.2);
    ctx.lineTo(x - radius, y);
    ctx.closePath();
  } else {
    ctx.moveTo(x - radius, y - radius);
    ctx.lineTo(x + radius, y + radius);
    ctx.moveTo(x + radius, y - radius);
    ctx.lineTo(x - radius, y + radius);
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = type === "cross" ? 6 : 3;
  if (type !== "cross") {
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
  }
  ctx.stroke();
}

function render(bins) {
  const width = 2148;
  const height = 1335;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  const margin = { left: 220, right: 80, top: 165, bottom: 195 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const observedRates = [...bins.values()].flatMap((modelBins) => modelBins.map(({ rate }) => rate)).filter(Number.isFinite);
  const maximumRate = Math.max(0, ...observedRates);
  const yStep = 0.005;
  const yMax = Math.max(0.01, Math.ceil((maximumRate * 1.2) / yStep) * yStep);
  const xScale = (value) => margin.left + (value / finalTimestep) * plotWidth;
  const yScale = (value) => margin.top + ((yMax - value) / yMax) * plotHeight;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.font = "40px Arial, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let tick = 0; tick <= yMax + 1e-12; tick += yStep) {
    const y = yScale(tick);
    ctx.strokeStyle = "#dddddd";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillStyle = "#111111";
    ctx.fillText(`${(tick * 100).toFixed(1)}%`, margin.left - 24, y);
  }

  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, height - margin.bottom);
  ctx.lineTo(width - margin.right, height - margin.bottom);
  ctx.stroke();

  ctx.font = "34px Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let index = 0; index < binCount; index += 1) {
    const center = index * binSize + binSize / 2;
    const x = xScale(center);
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, height - margin.bottom);
    ctx.lineTo(x, height - margin.bottom + 12);
    ctx.stroke();
    ctx.fillStyle = "#111111";
    ctx.fillText(`${index * 10}–${(index + 1) * 10}k`, x, height - margin.bottom + 25);
  }

  for (const model of modelOrder) {
    const modelBins = bins.get(model);
    ctx.strokeStyle = modelColors[model];
    ctx.lineWidth = 6;
    ctx.beginPath();
    let activeLine = false;
    for (const bin of modelBins) {
      if (!Number.isFinite(bin.rate)) {
        activeLine = false;
        continue;
      }
      const x = xScale(bin.center + modelOffsets[model]);
      const y = yScale(bin.rate);
      if (activeLine) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      activeLine = true;
    }
    ctx.stroke();
    for (const bin of modelBins) {
      if (!Number.isFinite(bin.rate)) continue;
      drawMarker(
        ctx,
        modelMarkers[model],
        xScale(bin.center + modelOffsets[model]),
        yScale(bin.rate),
        13,
        modelColors[model],
      );
    }
  }

  ctx.fillStyle = "#111111";
  ctx.font = "54px Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("LunarLander Training Success Rate by 10k-Step Window", width / 2, 78);
  ctx.font = "43px Arial, sans-serif";
  ctx.fillText("Environment interactions", margin.left + plotWidth / 2, height - 55);
  ctx.save();
  ctx.translate(58, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Training episode success rate", 0, 0);
  ctx.restore();

  const legendPositions = [
    [margin.left + 35, margin.top + 55],
    [margin.left + 330, margin.top + 55],
    [margin.left + 625, margin.top + 55],
    [margin.left + 35, margin.top + 115],
    [margin.left + 330, margin.top + 115],
  ];
  ctx.font = "34px Arial, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  modelOrder.forEach((model, index) => {
    const [x, y] = legendPositions[index];
    ctx.strokeStyle = modelColors[model];
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 62, y);
    ctx.stroke();
    drawMarker(ctx, modelMarkers[model], x + 31, y, 10, modelColors[model]);
    ctx.fillStyle = "#111111";
    ctx.fillText(model, x + 78, y);
  });

  fs.writeFileSync(outputPng, canvas.toBuffer("image/png"));
}

const { bins, finalEvaluations } = loadAndBin();
render(bins);
for (const model of modelOrder) {
  const modelBins = bins.get(model);
  const successes = modelBins.reduce((sum, bin) => sum + bin.successes, 0);
  const episodes = modelBins.reduce((sum, bin) => sum + bin.episodes, 0);
  console.log(`${model}: ${successes} successful training episodes / ${episodes} total episodes`);
}
console.log(`Saved ${path.basename(outputPng)}: 5 series, 10 bins each, ${finalEvaluations.length}/15 final evaluations verified at zero success`);
