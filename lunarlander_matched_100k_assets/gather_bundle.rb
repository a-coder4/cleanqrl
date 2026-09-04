#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "digest"
require "fileutils"
require "open3"

SOURCE_REF = ENV.fetch("SOURCE_REF", "origin/main")
OUTPUT_DIR = File.expand_path(__dir__)
EXPECTED_AGENTS = ["PPO", "PPO-tiny", "DQN", "QRL", "Quantum DQN"].freeze
EXPECTED_SEEDS = %w[0 1 2].freeze

FIGURES = {
  "01_reward_curves_matched_100k.png" =>
    "lunarlander_comparison_plots/reward_vs_timesteps.png",
  "02_final_reward_comparison.png" =>
    "lunarlander_comparison_plots/final_reward_distribution.png",
  "03_best_qrl_vs_classical.png" =>
    "lunarlander_comparison_plots/best_qrl_vs_classical_final_reward.png",
  "04_success_rate_comparison.png" =>
    "lunarlander_comparison_plots/success_rate.png",
  "05_compute_cost_combined.png" =>
    "lunarlander_comparison_plots/compute_cost_comparison.png",
  "05a_training_time_wall_clock.png" =>
    "final_report_figures/fig_compute_wall_clock_by_result_category.png",
  "05b_training_throughput_sps.png" =>
    "final_report_figures/fig_compute_sps_full_classical_vs_qppo_short.png",
  "05c_computational_cost_circuit_evaluations.png" =>
    "final_report_figures/fig_compute_circuit_evaluations_and_shots.png"
}.freeze

def git_output(*args)
  stdout, stderr, status = Open3.capture3("git", *args)
  abort(stderr) unless status.success?
  stdout
end

def git_blob(path)
  git_output("show", "#{SOURCE_REF}:#{path}")
end

def numeric(value)
  return nil if value.nil? || value.empty?

  Float(value)
end

commit = git_output("rev-parse", SOURCE_REF).strip
aggregate = CSV.parse(git_blob("lunarlander_aggregate_results.csv"), headers: true)
matched = aggregate.select { |row| row["included_in_plots"] == "yes" }
grouped = matched.group_by { |row| [row["agent_label"], row["seed"], row["run_name"]] }

actual_agent_seeds = grouped.keys.group_by(&:first).transform_values do |keys|
  keys.map { |key| key[1] }.uniq.sort
end
unless actual_agent_seeds.keys.sort == EXPECTED_AGENTS.sort
  abort("Unexpected matched agents: #{actual_agent_seeds.keys.sort.inspect}")
end
actual_agent_seeds.each do |agent, seeds|
  abort("Unexpected seeds for #{agent}: #{seeds.inspect}") unless seeds == EXPECTED_SEEDS
end

summary_rows = grouped.map do |(agent, seed, run_name), rows|
  max_timestep = rows.map { |row| row["timestep"].to_i }.max
  abort("#{run_name} ends at #{max_timestep}, not 100000") unless max_timestep == 100_000

  evaluation_rows = rows.select { |row| row["metric_type"] == "evaluation" }
  final_evaluation = evaluation_rows.max_by { |row| row["timestep"].to_i }
  abort("#{run_name} has no evaluation row") if final_evaluation.nil?

  latest_sps = rows.reject { |row| row["SPS"].to_s.empty? }.max_by { |row| row["timestep"].to_i }
  wall_clock = rows.filter_map { |row| numeric(row["wall_clock_time"]) }.max
  circuits = rows.filter_map { |row| numeric(row["circuit_evaluations"]) }.max
  metric_counts = rows.group_by { |row| row["metric_type"] }.transform_values(&:length)

  {
    "model" => agent,
    "agent_internal" => rows.first["agent"],
    "seed" => seed,
    "run_name" => run_name,
    "rows" => rows.length,
    "train_episode_rows" => metric_counts.fetch("train_episode", 0),
    "training_diagnostic_rows" => metric_counts.fetch("training_diagnostic", 0),
    "evaluation_rows" => metric_counts.fetch("evaluation", 0),
    "max_timestep" => max_timestep,
    "final_evaluation_reward" => final_evaluation["evaluation_reward"],
    "final_success_rate" => final_evaluation["success_rate"],
    "final_sps" => latest_sps&.fetch("SPS", nil),
    "wall_clock_time_seconds" => wall_clock,
    "circuit_evaluations" => circuits
  }
end.sort_by { |row| [EXPECTED_AGENTS.index(row["model"]), row["seed"].to_i] }

FileUtils.mkdir_p(OUTPUT_DIR)

CSV.open(File.join(OUTPUT_DIR, "matched_100k_all_metrics.csv"), "wb") do |csv|
  csv << aggregate.headers
  matched.each { |row| csv << row.fields }
end

seed_headers = summary_rows.first.keys
CSV.open(File.join(OUTPUT_DIR, "matched_100k_seed_report.csv"), "wb") do |csv|
  csv << seed_headers
  summary_rows.each { |row| csv << seed_headers.map { |header| row[header] } }
end

File.binwrite(
  File.join(OUTPUT_DIR, "matched_100k_model_summary.csv"),
  git_blob("compute_cost_comparison.csv")
)

FIGURES.each do |destination, source|
  File.binwrite(File.join(OUTPUT_DIR, destination), git_blob(source))
end

node_modules = ENV.fetch(
  "CLEANQRL_NODE_MODULES",
  File.join(Dir.home, ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules")
)
plot_stdout, plot_stderr, plot_status = Open3.capture3(
  { "NODE_PATH" => node_modules },
  "node",
  File.join(OUTPUT_DIR, "regenerate_figure_01b.cjs")
)
abort(plot_stderr) unless plot_status.success?
puts plot_stdout

readme = <<~MARKDOWN
  # LunarLander matched-100k asset bundle

  This flat folder gathers the newest matched-100k LunarLander data and figures available in this repository.

  - Source Git ref: `#{SOURCE_REF}`
  - Source commit: `#{commit}`
  - Protocol: 100,000 environment interactions per run
  - Models: PPO, PPO-tiny, DQN, QRL (the QPPO/QRL hybrid), and Quantum DQN
  - Seeds: 0, 1, and 2 for every model
  - Validated matched runs: #{summary_rows.length}
  - Matched metric rows: #{matched.length}

  ## Data

  - `matched_100k_all_metrics.csv`: all metric rows for the 15 included matched runs only. The source aggregate's older/excluded runs have been removed.
  - `matched_100k_seed_report.csv`: one row per model/seed, including the source run name, row counts, final evaluation reward/success rate, SPS, wall-clock time, and circuit evaluations.
  - `matched_100k_model_summary.csv`: five-model aggregate compute/performance summary used by the report figures.
  - `matched_100k_data.xlsx`: formatted workbook containing the full metrics, seed report, model summary, and source/protocol notes.

  ## Figures

  - `01_reward_curves_matched_100k.png`: rolling-mean training reward curves for all five models.
  - `01b_final_evaluation_rewards_at_100k_all_seeds.png`: categorical comparison of all three final evaluation points per model at 100k.
  - `02_final_reward_comparison.png`: final-reward distribution across the three seeds per model.
  - `03_best_qrl_vs_classical.png`: best QRL seed versus classical model means.
  - `04_success_rate_comparison.png`: success rate over environment interactions.
  - `05_compute_cost_combined.png`: combined SPS and circuit-evaluation comparison.
  - `05a_training_time_wall_clock.png`: mean wall-clock training time.
  - `05b_training_throughput_sps.png`: mean steps per second.
  - `05c_computational_cost_circuit_evaluations.png`: mean circuit-evaluation cost.

  ## Interpretation notes

  `QRL` is the repository's label for the QPPO/QRL hybrid (`qppo_hybrid_configC`). Final evaluation success is 0 for all 15 runs at 100k steps. The success-rate curve can still contain isolated successful training episodes before the final evaluation. The final-reward distribution uses each run's last-100 training-episode mean, while the seed report and best-QRL bar use the final evaluation reward at 100k.

  **Figure 2 caption:** Final LunarLander-v3 evaluation rewards at 100k steps for all three seeds of each matched model. Each point represents one seed; the dashed line marks the solved threshold.

  `gather_bundle.rb` reproduces the repository-sourced CSV and figure files from the source Git ref, regenerates Figure 2 with `regenerate_figure_01b.cjs`, and validates the five-model, three-seed, 100k protocol before writing outputs.
MARKDOWN
File.write(File.join(OUTPUT_DIR, "README.md"), readme)

checksum_files = Dir.children(OUTPUT_DIR).reject do |name|
  name == "SHA256SUMS.txt" || File.directory?(File.join(OUTPUT_DIR, name))
end.sort
checksums = checksum_files.map do |name|
  "#{Digest::SHA256.file(File.join(OUTPUT_DIR, name)).hexdigest}  #{name}"
end
File.write(File.join(OUTPUT_DIR, "SHA256SUMS.txt"), checksums.join("\n") + "\n")

puts "Created #{OUTPUT_DIR}"
puts "Validated #{summary_rows.length} runs and #{matched.length} metric rows from #{commit}"
