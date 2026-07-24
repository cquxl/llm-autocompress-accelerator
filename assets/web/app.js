const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;"
})[char]);
const number = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? "—" : Number(value).toFixed(digits);

function generationHtml(generation) {
  if (!generation || !generation.prompt) return "";
  const streaming = generation.status === "streaming";
  return `
    <section class="generation">
      <div class="row">
        <strong>实时推理 · ${escapeHtml(generation.backend || "")}</strong>
        <span class="live">${streaming ? "● GENERATING" : "✓ COMPLETED"}</span>
      </div>
      <p class="prompt"><b>Prompt</b>　${escapeHtml(generation.prompt)}</p>
      <div class="output">${escapeHtml(generation.text || "等待首个 token…")}${streaming ? `<span class="cursor">▋</span>` : ""}</div>
      <div class="metrics">
        <span>${number(generation.tokens, 0)} tokens</span>
        <span>${number(generation.tokens_per_second)} tokens/s</span>
        <span>${number(generation.elapsed_seconds)} s</span>
      </div>
    </section>`;
}

function candidateResultsHtml(results) {
  if (!results?.length) return "";
  const rows = results.map(item => {
    const candidate = item.candidate || {};
    const metrics = item.metrics || {};
    const artifact = item.artifact || {};
    return `<tr>
      <td>${escapeHtml(candidate.id || "")}</td>
      <td>${escapeHtml(candidate.algorithm || "")}</td>
      <td>${escapeHtml(candidate.backend || "")}</td>
      <td>${escapeHtml(item.status || "")}</td>
      <td>${artifact.output_bytes ? number(artifact.output_bytes / 1024 / 1024) + " MiB" : "—"}</td>
      <td>${number(artifact.compression_ratio, 3)}×</td>
      <td>${number(artifact.effective_weight_compression_ratio, 2)}×</td>
      <td>${artifact.sparsity_after == null ? "—" : number(artifact.sparsity_after * 100) + "%"}</td>
      <td>${number(item.quality?.perplexity, 3)}</td>
      <td>${number(metrics.prefill_tokens_per_second)}</td>
      <td>${number(metrics.decode_tokens_per_second)}</td>
      <td>${number(metrics.ttft_ms)}</td>
    </tr>`;
  }).join("");
  return `<div class="table-wrap"><table>
    <thead><tr><th>候选</th><th>算法</th><th>后端</th><th>状态</th><th>Artifact</th><th>Checkpoint大小比</th><th>有效权重压缩</th><th>稀疏率</th><th>PPL</th><th>Prefill tok/s</th><th>Decode tok/s</th><th>TTFT ms</th></tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`;
}

function evaluationHtml(evaluation) {
  if (!evaluation) return "";
  const baseline = evaluation.baseline || {};
  const tf = baseline.transformers || {};
  const vllm = baseline.vllm || {};
  const compressed = (evaluation.evaluated || []).length > 0;
  const recommended = evaluation.recommended?.id || "无候选通过全部质量、压缩与速度门槛";
  const trace = evaluation.search_trace || {};
  const decisions = trace.decisions || [];
  const traceRows = decisions.map(item => `<tr>
    <td>${number(item.trial, 0)}</td>
    <td>${escapeHtml(item.candidate || "")}</td>
    <td>${item.quality_pass ? "通过" : "未通过"}</td>
    <td>${item.compression_pass ? "通过" : "未通过"}</td>
    <td>${item.speed_pass ? "通过" : "未通过"}</td>
    <td>${item.business_target_met ? "达标，提前停止" : "继续搜索"}</td>
  </tr>`).join("");
  const rows = (evaluation.evaluated || []).map(item => `<tr>
    <td>${escapeHtml(item.id)}</td>
    <td>${escapeHtml(item.execution_status)}</td>
    <td>${item.artifact_bytes ? number(item.artifact_bytes / 1024 / 1024) + " MiB" : "—"}</td>
    <td>${number(item.compression_ratio, 3)}×</td>
    <td>${number(item.effective_weight_compression_ratio, 2)}×</td>
    <td>${item.sparsity_after == null ? "—" : number(item.sparsity_after * 100) + "%"}</td>
    <td>${number(item.perplexity, 3)}</td>
    <td>${item.relative_perplexity_increase == null ? "—" : number(item.relative_perplexity_increase * 100) + "%"}</td>
    <td>${number(item.same_backend_speedup, 3)}×</td>
    <td>${number(item.deployment_speedup, 3)}×</td>
    <td>${number(item.micro_kernel_speedup, 3)}×</td>
    <td>${item.accepted ? "通过" : "未通过"}</td>
  </tr>`).join("");
  return `<section class="evaluation">
    <h3>${compressed ? "自动压缩最终报告" : "Dense Baseline 验证报告（未压缩）"}</h3>
    <p><b>推荐：</b>${escapeHtml(recommended)}</p>
    ${trace.strategy?.enabled ? `<div class="search-decision">
      <p><b>自动搜索结论：</b>${trace.business_target_met
        ? `业务目标已满足，选择 ${escapeHtml(trace.selected_candidate || "")} 并提前停止`
        : "试验预算内没有候选满足全部业务门槛"}</p>
      <p>实际压缩试验 ${number(trace.trials_executed, 0)} / ${number(trace.strategy.max_trials, 0)}；
      剪枝回退：${trace.pruning_fallback_triggered ? "已触发" : "无需触发"}</p>
      ${traceRows ? `<div class="table-wrap"><table>
        <thead><tr><th>试验</th><th>候选</th><th>质量</th><th>压缩</th><th>速度</th><th>决策</th></tr></thead>
        <tbody>${traceRows}</tbody>
      </table></div>` : ""}
    </div>` : ""}
    <div class="metrics">
      <span>Baseline PPL ${number(baseline.perplexity, 3)}</span>
      <span>Prefill ${number(tf.prefill_tokens_per_second)} tok/s</span>
      <span>Decode ${number(tf.decode_tokens_per_second)} tok/s</span>
      <span>TTFT ${number(tf.ttft_ms)} ms</span>
      <span>TPOT ${number(tf.tpot_ms)} ms</span>
      <span>vLLM Decode ${number(vllm.decode_tokens_per_second)} tok/s</span>
      <span>vLLM E2E ${number(vllm.end_to_end_ms)} ms</span>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>候选</th><th>状态</th><th>Artifact</th><th>Checkpoint大小比</th><th>有效权重压缩</th><th>稀疏率</th><th>PPL</th><th>PPL Δ</th><th>同后端</th><th>部署收益</th><th>Kernel</th><th>质量门控</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>`;
}
const payload = () => ({
  model: $("model").value,
  profile: $("profile").value,
  preset: $("preset").value,
  target_checkpoint_ratio: Number($("targetRatio").value),
  max_relative_ppl_increase: Number($("maxPplIncrease").value),
  min_same_backend_speedup: Number($("minSpeedup").value),
  pruning_granularity: $("pruningGranularity").value,
  prompt: $("prompt").value
});

async function preview() {
  const response = await fetch("/api/bootstrap", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload())
  });
  $("strategy").textContent = JSON.stringify(await response.json(), null, 2);
}

async function run() {
  $("run").disabled = true;
  $("run").textContent = "正在创建任务…";
  try {
    const response = await fetch("/api/jobs", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...payload(), mode: "smoke", yes: true})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    $("strategy").textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    $("strategy").textContent = `启动失败：${error.message}`;
    $("run").disabled = false;
    $("run").textContent = "运行自动压缩 Demo";
  }
  await refresh();
}

async function cancel(id) {
  await fetch(`/api/jobs/${id}/cancel`, {method: "POST"});
  refresh();
}

let refreshTimer = null;
let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  let nextRefreshMs = 3000;
  try {
    const response = await fetch("/api/jobs", {cache: "no-store"});
    const data = await response.json();
    const busy = data.jobs.some(job => ["queued", "running"].includes(job.status));
    const streaming = data.jobs.some(job => job.live_generation?.status === "streaming");
    nextRefreshMs = streaming ? 100 : busy ? 750 : 3000;
    $("run").disabled = busy;
    $("run").textContent = busy ? "任务执行中…" : "运行自动压缩 Demo";
    $("jobs").innerHTML = data.jobs.slice().reverse().map(job => `
    <article class="job">
      <div class="row"><strong>${escapeHtml(job.id)}</strong><span class="status ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></div>
      <p class="step">${escapeHtml(job.current_step || "等待任务状态")}</p>
      <span class="kind ${escapeHtml(job.task_kind)}">${job.task_kind === "automatic_compression" ? "自动压缩对比" : job.task_kind === "dense_baseline_validation" ? "仅 Dense 验证" : "准备中"}</span>
      ${job.current_candidate ? `<p class="pipeline">当前：<b>${escapeHtml(job.current_candidate.algorithm)}</b> · ${escapeHtml(job.current_candidate.structure)} → <b>${escapeHtml(job.current_candidate.backend)}</b></p>` : ""}
      ${(job.progress?.total || 0) > 0 ? `
        <progress value="${job.progress.current}" max="${job.progress.total}"></progress>
        <small>${job.progress.current} / ${job.progress.total} 个候选</small>
      ` : `<progress class="indeterminate"></progress>`}
      <p class="error">${escapeHtml(job.error || "")}</p>
      ${job.run_dir ? `<p class="run-dir">${escapeHtml(job.run_dir)}</p>` : ""}
      ${generationHtml(job.live_generation)}
      ${candidateResultsHtml(job.candidate_results)}
      ${evaluationHtml(job.evaluation)}
      ${job.status === "completed" ? `<p>
        <a href="/api/artifacts/${job.id}/report.md" target="_blank">查看报告</a> ·
        <a href="/api/artifacts/${job.id}/evaluation.json" target="_blank">评测 JSON</a> ·
        <a href="/api/artifacts/${job.id}/results.csv" target="_blank">结果 CSV</a>
      </p><img class="chart" src="/api/artifacts/${job.id}/charts/speedup.svg" alt="候选加速结果">` : ""}
      ${["queued","running"].includes(job.status) ? `<button onclick="cancel('${job.id}')">取消</button>` : ""}
      <details ${job.status === "failed" ? "open" : ""}><summary>执行日志</summary><pre>${escapeHtml((job.logs || []).slice(-30).join("\n"))}</pre></details>
    </article>`).join("") || "<p>暂无任务。</p>";
  } catch (error) {
    $("jobs").innerHTML = `<p class="error">状态刷新失败：${escapeHtml(error.message)}</p>`;
  } finally {
    refreshing = false;
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, nextRefreshMs);
  }
}

$("preview").onclick = preview;
$("run").onclick = run;
$("refresh").onclick = refresh;
$("preset").onchange = preview;
$("model").onchange = preview;
$("targetRatio").onchange = preview;
$("maxPplIncrease").onchange = preview;
$("minSpeedup").onchange = preview;
$("pruningGranularity").onchange = preview;
preview();
refresh();
