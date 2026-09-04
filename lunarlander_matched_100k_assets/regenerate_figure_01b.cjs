#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { createCanvas } = require("@napi-rs/canvas");

const bundleDir = __dirname;
const sourceCsv = path.join(bundleDir, "matched_100k_seed_report.csv");
const outputPng = path.join(bundleDir, "01b_final_evaluation_rewards_at_100k_all_seeds.png");

const modelOrder = ["PPO", "PPO-tiny", "DQN", "QRL", "Quantum DQN"];
const modelColors = {
  PPO: "#4169e1",
  "PPO-tiny": "#00a676",
  DQN: "#d95f02",
  QRL: "#7b3294",
  "Quantum DQN": "#b35806",
};
const seedOffsets = new Map([[0, -0.10], [1, 0.00], [2, 0.10]]);
const seedMarkers = new Map([[0, "circle"], [1, "square"], [2, "triangle"]]);

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

function loadAndValidate() {
  const rows = parseCsv(fs.readFileSync(sourceCsv, "utf8"));
  if (rows.length !== 15) throw new Error(`Expected 15 seed rows, found ${rows.length}`);

  const grouped = new Map(modelOrder.map((model) => [model, []]));
  for (const row of rows) {
    const model = row.model;
    const seed = Number(row.seed);
    const reward = Number(row.final_evaluation_reward);
    const maxTimestep = Number(row.max_timestep);
    const evaluationRows = Number(row.evaluation_rows);

    if (!grouped.has(model)) throw new Error(`Unexpected model: ${model}`);
    if (!seedOffsets.has(seed)) throw new Error(`Unexpected seed for ${model}: ${seed}`);
    if (maxTimestep !== 100000) throw new Error(`${model} seed ${seed} ends at ${maxTimestep}, not 100000`);
    if (evaluationRows !== 1) throw new Error(`${model} seed ${seed} has ${evaluationRows} evaluation rows, expected 1`);
    if (!Number.isFinite(reward)) throw new Error(`Non-finite reward for ${model} seed ${seed}: ${reward}`);
    grouped.get(model).push({ seed, reward });
  }

  for (const model of modelOrder) {
    grouped.get(model).sort((a, b) => a.seed - b.seed);
    const seeds = grouped.get(model).map(({ seed }) => seed);
    if (seeds.join(",") !== "0,1,2") throw new Error(`Expected seeds 0,1,2 for ${model}, found ${seeds}`);
  }
  return grouped;
}

function drawMarker(ctx, type, x, y, radius, fill) {
  ctx.beginPath();
  if (type === "circle") {
    ctx.arc(x, y, radius, 0, Math.PI * 2);
  } else if (type === "square") {
    ctx.rect(x - radius, y - radius, radius * 2, radius * 2);
  } else {
    ctx.moveTo(x, y - radius * 1.18);
    ctx.lineTo(x + radius * 1.08, y + radius * 0.92);
    ctx.lineTo(x - radius * 1.08, y + radius * 0.92);
    ctx.closePath();
  }
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  ctx.stroke();
}

function render(grouped) {
  const width = 2148;
  const height = 1335;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext("2d");
  const margin = { left: 230, right: 80, top: 155, bottom: 190 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yMin = -2000;
  const yMax = 300;

  const xScale = (categoryPosition) => margin.left + ((categoryPosition + 0.5) / modelOrder.length) * plotWidth;
  const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.font = "42px Arial, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let tick = -2000; tick <= 250; tick += 250) {
    const y = yScale(tick);
    ctx.strokeStyle = "#dddddd";
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillStyle = "#111111";
    ctx.fillText(tick.toLocaleString("en-US").replace("-", "−"), margin.left - 24, y);
  }

  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, height - margin.bottom);
  ctx.lineTo(width - margin.right, height - margin.bottom);
  ctx.stroke();

  ctx.font = "42px Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  modelOrder.forEach((model, index) => {
    const x = xScale(index);
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x, height - margin.bottom);
    ctx.lineTo(x, height - margin.bottom + 14);
    ctx.stroke();
    ctx.fillStyle = "#111111";
    ctx.fillText(model, x, height - margin.bottom + 28);
  });

  let plottedPoints = 0;
  modelOrder.forEach((model, modelPosition) => {
    for (const { seed, reward } of grouped.get(model)) {
      drawMarker(
        ctx,
        seedMarkers.get(seed),
        xScale(modelPosition + seedOffsets.get(seed)),
        yScale(reward),
        17,
        modelColors[model],
      );
      plottedPoints += 1;
    }
  });
  if (plottedPoints !== 15) throw new Error(`Expected to plot 15 points, plotted ${plottedPoints}`);

  const thresholdY = yScale(200);
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 4;
  ctx.setLineDash([18, 12]);
  ctx.beginPath();
  ctx.moveTo(margin.left, thresholdY);
  ctx.lineTo(width - margin.right, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#111111";
  ctx.font = "56px Arial, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Matched 100k Final Evaluation Reward", width / 2, 74);
  ctx.font = "45px Arial, sans-serif";
  ctx.fillText("Agent", margin.left + plotWidth / 2, height - 55);

  ctx.save();
  ctx.translate(60, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Final evaluation reward", 0, 0);
  ctx.restore();

  const legendX = margin.left + 35;
  const legendY = margin.top + 70;
  const legendItems = [
    { marker: "circle", label: "Seed 0" },
    { marker: "square", label: "Seed 1" },
    { marker: "triangle", label: "Seed 2" },
  ];
  ctx.font = "36px Arial, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  legendItems.forEach((item, index) => {
    const column = index % 2;
    const row = Math.floor(index / 2);
    const x = legendX + column * 230;
    const y = legendY + row * 62;
    drawMarker(ctx, item.marker, x, y, 12, "#666666");
    ctx.fillStyle = "#111111";
    ctx.fillText(item.label, x + 30, y);
  });
  const thresholdLegendX = legendX + 230;
  const thresholdLegendY = legendY + 62;
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 4;
  ctx.setLineDash([16, 10]);
  ctx.beginPath();
  ctx.moveTo(thresholdLegendX - 12, thresholdLegendY);
  ctx.lineTo(thresholdLegendX + 28, thresholdLegendY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#111111";
  ctx.fillText("Solved threshold", thresholdLegendX + 46, thresholdLegendY);

  fs.writeFileSync(outputPng, canvas.toBuffer("image/png"));
  return plottedPoints;
}

const grouped = loadAndValidate();
const plottedPoints = render(grouped);
console.log(`Saved ${path.basename(outputPng)}: ${grouped.size} models, ${plottedPoints} seed points`);
