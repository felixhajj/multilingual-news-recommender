const state = {
  users: [],
  articles: [],
  userId: "user_004",
  articleId: "incoming_001",
  notebookSteps: null,
};

const categoryLabels = {
  countries: "Countries",
  companies: "Companies",
  organizations: "Organisations",
  profiles: "Profiles",
  systems: "Systems",
  topics: "Topics",
};

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error || `Request failed: ${url}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function prettyJson(value) {
  return escapeHtml(JSON.stringify(value, null, 2));
}

function notebookCodePanel(stageId) {
  const cells = state.notebookSteps?.stages?.[stageId] || [];
  if (cells.length === 0) {
    return "";
  }

  const notebookName = state.notebookSteps.notebook;
  return `
    <details class="notebook-code">
      <summary>
        <span>View the real notebook code for this step</span>
        <small>${escapeHtml(notebookName)}</small>
      </summary>
      <div class="notebook-code-body">
        ${cells.map((cell) => `
          <section class="notebook-cell">
            <div class="notebook-cell-header">
              <span>Code cell ${cell.cell_index}</span>
              <span>Execution ${cell.execution_count ?? "not run"}</span>
            </div>
            <pre><code>${escapeHtml(cell.source)}</code></pre>
          </section>
        `).join("")}
      </div>
    </details>
  `;
}

function chips(values, emptyLabel = "None") {
  if (!values || values.length === 0) {
    return `<span class="chip muted">${emptyLabel}</span>`;
  }
  return values.map((value) => `<span class="chip">${escapeHtml(value)}</span>`).join("");
}

function tagGroups(tags) {
  return Object.entries(categoryLabels)
    .map(([key, label]) => `
      <div class="tag-group">
        <span>${label}</span>
        <div>${chips(tags?.[key])}</div>
      </div>
    `)
    .join("");
}

function metric(label, before, after, asPercent = true) {
  const beforeValue = asPercent ? `${Math.round(before * 100)}%` : before;
  const afterValue = asPercent ? `${Math.round(after * 100)}%` : after;
  const width = Math.max(4, Math.round(after * 100));
  return `
    <div class="progress-metric">
      <div><span>${label}</span><strong>${beforeValue} → ${afterValue}</strong></div>
      <div class="progress-track"><i style="width:${width}%"></i></div>
    </div>
  `;
}

function tokenPreview(stage) {
  if (stage.status !== "ready") {
    return `<p class="warning">${escapeHtml(stage.error)}</p>`;
  }
  return `
    <div class="token-grid">
      ${stage.preview.map((item) => `
        <div class="token">
          <span>${escapeHtml(item.decoded || item.token)}</span>
          <code>${item.id}</code>
        </div>
      `).join("")}
    </div>
    <p class="caption">Showing ${stage.preview.length} of ${stage.token_count} token IDs. Vocabulary size: ${stage.vocabulary_size.toLocaleString()}.</p>
  `;
}

function filterValueChips(items, side) {
  if (!items?.length) {
    return '<span class="filter-value empty">None</span>';
  }
  return items.map((item) => {
    const stateClass = item.matched ? "shared" : side === "user" ? "missing" : "extra";
    return `<span class="filter-value ${stateClass}">${escapeHtml(item.value)}</span>`;
  }).join("");
}

function renderPipeline(payload) {
  const input = payload.input;
  const progress = payload.training_progress;
  const extraction = payload.qwen_extraction;
  const entities = payload.entity_enrichment;
  const structured = payload.structured_result;
  const embeddings = payload.embeddings;
  const score = payload.scoring;

  const entityRows = entities.matched_entities.length
    ? entities.matched_entities.map((entity) => `
        <tr>
          <td>${escapeHtml(entity.matched_aliases?.join(", ") || entity.canonical_name)}</td>
          <td>→</td>
          <td><strong>${escapeHtml(entity.canonical_name)}</strong><small>${escapeHtml(entity.entity_id)}</small></td>
        </tr>
      `).join("")
    : '<tr><td colspan="3">No catalogue entity was linked for this article.</td></tr>';

  const filterRows = structured.filter_comparison.map((row) => `
    <div class="filter-compare-row ${row.matched ? "matched" : "unmatched"}">
      <strong class="filter-category">${escapeHtml(categoryLabels[row.category])}</strong>
      <div class="filter-values">${filterValueChips(row.user_values, "user")}</div>
      <span class="filter-status">${row.matched ? "Match" : "Missing"}</span>
      <div class="filter-values">${filterValueChips(row.article_values, "article")}</div>
    </div>
  `).join("");

  const semanticLinks = embeddings.concept_links.length
    ? embeddings.concept_links.map((link) => `
        <div class="concept-link">
          <strong>${escapeHtml(link.user_phrase)}</strong>
          <span>E5 phrase similarity · ${Math.round(link.similarity * 100)}%</span>
          <strong dir="rtl">${escapeHtml(link.article_phrase)}</strong>
        </div>
      `).join("")
    : '<p class="caption">E5 still scores the full texts, but no short phrase example is available for this selection.</p>';

  document.querySelector("#pipeline").innerHTML = `
    <article class="stage-card input-stage">
      <div class="stage-number">01</div>
      <div class="stage-content">
        <p class="eyebrow">Application input</p>
        <h2>One article + one user profile</h2>
        <div class="two-columns">
          <div>
            <h3>Article</h3>
            <p class="article-title" dir="auto">${escapeHtml(input.article.title)}</p>
            <p dir="auto">${escapeHtml(input.article.summary || "No summary")}</p>
          </div>
          <div>
            <h3>User profile</h3>
            <p class="article-title">${escapeHtml(input.user.name)}</p>
            <span class="field-label">User's explicit interest statement</span>
            <p class="interest-quote">${escapeHtml(input.user.query)}</p>
            <p class="caption">This sentence belongs to the user. E5 analyzes it later as a separate semantic signal.</p>
          </div>
        </div>
        ${notebookCodePanel("01")}
      </div>
    </article>

    <article class="stage-card">
      <div class="stage-number">02</div>
      <div class="stage-content">
        <p class="eyebrow">Qwen tokenizer</p>
        <h2>Text becomes token IDs</h2>
        <p class="stage-intro">These numbers are the input Qwen actually receives.</p>
        ${tokenPreview(payload.tokenizer)}
        ${notebookCodePanel("02")}
      </div>
    </article>

    <article class="stage-card">
      <div class="stage-number">03</div>
      <div class="stage-content">
        <p class="eyebrow">Qwen2.5-3B + QLoRA</p>
        <h2>The adapter learned the extraction format</h2>
        <div class="adapter-facts">
          <div><span>LoRA parameters</span><strong>${payload.adapter.parameter_count?.toLocaleString() || "Unavailable"}</strong></div>
          <div><span>Adapter size</span><strong>${payload.adapter.adapter_size_mb} MB</strong></div>
        </div>
        <div class="progress-grid">
          ${metric("Entity micro-F1", progress.baseline.micro_f1, progress.trained.micro_f1)}
          ${metric("Valid JSON", progress.baseline.json_validity, progress.trained.json_validity)}
        </div>
        <p class="caption">${escapeHtml(progress.note)}</p>
        <div class="output-label">
          <span>Saved Qwen + LoRA output</span>
          <span class="status-pill">${escapeHtml(extraction.status.replaceAll("_", " "))}</span>
        </div>
        ${extraction.raw_model_output
          ? `<pre><code>${escapeHtml(extraction.raw_model_output)}</code></pre>`
          : `<p class="warning">${escapeHtml(extraction.note)}</p>`}
        ${notebookCodePanel("03")}
      </div>
    </article>

    <article class="stage-card">
      <div class="stage-number">04</div>
      <div class="stage-content">
        <p class="eyebrow">Entity enrichment</p>
        <h2>Mentions become canonical filters</h2>
        <table class="entity-table">
          <thead><tr><th>Article mention</th><th></th><th>Canonical entity</th></tr></thead>
          <tbody>${entityRows}</tbody>
        </table>
        <div class="tag-groups">${tagGroups(entities.canonical_tags)}</div>
        <p class="caption">${escapeHtml(entities.note)}</p>
        <div class="linking-roadmap">
          <div>
            <span>Current prototype</span>
            <strong>Manually configured matching</strong>
            <p>Known aliases are matched to a curated entity catalogue.</p>
          </div>
          <b>to</b>
          <div>
            <span>Future version</span>
            <strong>Learned entity linker</strong>
            <p>A model will propose and score entity matches, with uncertain cases reviewed by people.</p>
          </div>
        </div>
        ${notebookCodePanel("04")}
      </div>
    </article>

    <article class="stage-card model-one-stage">
      <div class="stage-number">05</div>
      <div class="stage-content">
        <p class="eyebrow">Model 1 · Qwen2.5-3B + LoRA</p>
        <h2>Qwen creates filters for the article</h2>
        <p class="stage-intro">The model reads the article and outputs structured facts. Here they are compared with the user's required filters.</p>
        <div class="filter-comparison">
          <div class="filter-compare-header">
            <span>Filter</span>
            <strong>User requires</strong>
            <span></span>
            <strong>Article has</strong>
          </div>
          ${filterRows}
        </div>
        <div class="model-result ${structured.direct_match ? "success" : "partial"}">
          <span>Model 1 result</span>
          <strong>${structured.matched_filter_categories} of ${structured.total_filter_categories} required filters matched</strong>
          <p>${structured.direct_match ? "This article satisfies every required filter for this user." : "This article is missing at least one required filter."}</p>
        </div>
        <p class="caption">${escapeHtml(structured.note)}</p>
        ${notebookCodePanel("05")}
      </div>
    </article>

    <section class="model-handoff" aria-label="Model 2 starts here">
      <span><b>Model 1 · Qwen + LoRA</b><small>Article filters complete</small></span>
      <strong>Next</strong>
      <span><b>Model 2 · Multilingual E5</b><small>Compare meaning across languages</small></span>
    </section>

    <article class="stage-card e5-stage">
      <div class="stage-number">06</div>
      <div class="stage-content">
        <p class="eyebrow">Model 2 · Multilingual E5</p>
        <h2>E5 connects the English interest to the Arabic article</h2>
        <p class="stage-intro">E5 maps both languages directly into the same ${embeddings.vector_dimension}-number meaning space. It does not translate the article or use the entity catalogue.</p>
        <div class="semantic-inputs">
          <div>
            <span>User interest · English</span>
            <p>${escapeHtml(input.user.query)}</p>
          </div>
          <div>
            <span>Article · Arabic</span>
            <p dir="rtl">${escapeHtml(input.article.summary || input.article.title)}</p>
          </div>
        </div>
        <div class="concept-links">
          <p class="eyebrow">Phrase relationships found by E5</p>
          ${semanticLinks}
          <small>E5 finds these at runtime by embedding short phrases from both inputs and comparing them. The final 79% result still comes from comparing the complete sentences.</small>
        </div>
        <div class="model-result e5-result">
          <span>Model 2 result</span>
          <strong>${score.embedding_score}% meaning similarity</strong>
          <p>The article is relevant even though the interest is English and the article is Arabic.</p>
        </div>
        <details>
          <summary>Show the exact text sent to E5</summary>
          <div class="two-columns code-inputs">
            <div><strong>User-interest input</strong><pre>${escapeHtml(embeddings.user_input)}</pre></div>
            <div><strong>Natural article-text input</strong><pre>${escapeHtml(embeddings.article_input)}</pre></div>
          </div>
        </details>
        ${notebookCodePanel("06")}
      </div>
    </article>

    <article class="stage-card final-stage">
      <div class="stage-number">07</div>
      <div class="stage-content">
        <p class="eyebrow">Final ranking result</p>
        <h2>Combine the two model results</h2>
        <div class="score-formula">
          <div><span>Model 1 · Qwen + LoRA</span><strong>${structured.matched_filter_categories}/${structured.total_filter_categories}</strong><small>required filters matched · 25% weight</small></div>
          <b>+</b>
          <div><span>Model 2 · E5</span><strong>${score.embedding_score}%</strong><small>meaning similarity · 75% weight</small></div>
          <b>=</b>
          <div class="total"><span>Recommendation score</span><strong>${score.score}%</strong><small>${score.direct_match ? "All required filters matched" : "Related result"}</small></div>
        </div>
        ${notebookCodePanel("07")}
      </div>
    </article>
  `;
}

function renderRanking(payload) {
  const top = payload.recommendations.slice(0, 5);
  document.querySelector("#rankingSummary").textContent =
    `${payload.total_articles} articles compared · top ${top.length} shown`;
  document.querySelector("#ranking").innerHTML = top.map((item, index) => `
    <article class="rank-row ${item.article.article_id === state.articleId ? "selected" : ""}">
      <span class="rank">${String(index + 1).padStart(2, "0")}</span>
      <div>
        <h3>${escapeHtml(item.article.title)}</h3>
        <p>${item.direct_match ? "Direct required-filter match" : "Semantically related"} ·
          E5 ${item.embedding_score} · exact ${item.metadata_score}</p>
      </div>
      <strong>${item.score}%</strong>
    </article>
  `).join("");
}

async function runPipeline() {
  const button = document.querySelector("#runButton");
  const status = document.querySelector("#status");
  button.disabled = true;
  status.textContent = "Loading tokenizer, embeddings, and saved model artifacts…";

  try {
    const query = `user_id=${encodeURIComponent(state.userId)}&article_id=${encodeURIComponent(state.articleId)}`;
    const [pipeline, ranking] = await Promise.all([
      fetchJson(`/api/pipeline-demo?${query}`),
      fetchJson(`/api/recommendations?user_id=${encodeURIComponent(state.userId)}`),
    ]);
    renderPipeline(pipeline);
    renderRanking(ranking);
    status.textContent = "Pipeline complete. Every stage below uses the selected article and profile.";
  } catch (error) {
    status.textContent = error.message;
    document.querySelector("#pipeline").innerHTML =
      `<div class="error-card">${escapeHtml(error.message)}</div>`;
  } finally {
    button.disabled = false;
  }
}

async function init() {
  const [users, articles, notebookSteps] = await Promise.all([
    fetchJson("/api/users"),
    fetchJson("/api/articles"),
    fetchJson("/api/notebook-steps"),
  ]);
  state.users = users;
  state.articles = articles;
  state.notebookSteps = notebookSteps;

  const userSelect = document.querySelector("#userSelect");
  userSelect.innerHTML = users
    .map((user) => `<option value="${user.user_id}">${escapeHtml(user.name)}</option>`)
    .join("");
  userSelect.value = state.userId;

  const articleSelect = document.querySelector("#articleSelect");
  articleSelect.innerHTML = articles
    .map((article) => {
      const marker = article.article_id.startsWith("incoming_") ? "Incoming · " : "Archive · ";
      return `<option value="${article.article_id}">${marker}${escapeHtml(article.title)}</option>`;
    })
    .join("");
  articleSelect.value = state.articleId;

  userSelect.addEventListener("change", (event) => {
    state.userId = event.target.value;
  });
  articleSelect.addEventListener("change", (event) => {
    state.articleId = event.target.value;
  });
  document.querySelector("#runButton").addEventListener("click", runPipeline);

  await runPipeline();
}

init().catch((error) => {
  document.querySelector("#status").textContent = error.message;
});
