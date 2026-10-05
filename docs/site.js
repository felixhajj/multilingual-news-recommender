const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
async function load() {
  const [ranking, evidence, deployment] = await Promise.all(
    ["example_ranking.json", "evidence.json", "deployment.json"].map(async file => {
      const response = await fetch(file);
      if (!response.ok) throw new Error(`Cannot load ${file}`);
      return response.json();
    })
  );
  document.querySelector("#interest").textContent = `Recorded interest: ${ranking.interest || "not prepared"}`;
  const results = document.querySelector("#results");
  results.replaceChildren();
  for (const [index, row] of (ranking.results || []).slice(0, 3).entries()) {
    const article = row.article;
    const card = element("article", undefined, "record");
    card.append(element("span", String(index + 1).padStart(2, "0"), "position"));
    const content = element("div");
    content.append(element("small", `${article.language || ""} / ${article.date || "date unavailable"} / recorded score ${row.score.toFixed(3)}`));
    const heading = element("h3", article.title); heading.dir = "auto"; content.append(heading);
    const paragraph = element("p", article.body.slice(0, 300) + "..."); paragraph.dir = "auto"; content.append(paragraph);
    if (/^https:\/\//.test(article.source_url || "")) {
      const link = element("a", "Original source and contributors"); link.href = article.source_url; content.append(link);
    }
    content.append(element("p", `${article.author || "Source contributors"} / ${article.license_id || "See source license"}. Cleaned excerpt.`, "small"));
    card.append(content); results.append(card);
  }
  if (!(ranking.results || []).length) results.append(element("p", "No completed inference examples have been exported yet."));
  const metrics = evidence.linker?.overall;
  const values = [
    [evidence.corpus?.documents ?? "Pending", "cleaned historical articles"],
    [evidence.release?.articles ?? 0, "articles processed in the recorded run"],
    [metrics ? `${(metrics.accepted_precision * 100).toFixed(1)}%` : "Pending", metrics ? `accepted link precision; ${(metrics.coverage * 100).toFixed(1)}% coverage` : "final linker evaluation"]
  ];
  const numbers = document.querySelector("#numbers");
  for (const [value, label] of values) { const box = element("div", undefined, "number"); box.append(element("strong", String(value)), element("small", label)); numbers.append(box); }
  document.querySelector("#status").textContent = `Extraction review: ${evidence.review?.human_reviewed ?? 0}/${evidence.review?.total ?? 30}. Active extraction run: ${evidence.active_model?.run_id || "not recorded"}. Pending review is not claimed as finished evaluation.`;
  const live = document.querySelector("#live");
  if (deployment.live_demo && /^https:\/\/huggingface.co\/spaces\//.test(deployment.live_demo)) {
    live.href = deployment.live_demo; live.textContent = "Try the live models";
    document.querySelector("#deployment").textContent = "Free hosted inference: startup delays and daily quotas apply.";
  } else document.querySelector("#deployment").textContent = "Research preview. Live publication awaits review, validation and account setup.";
}
load().catch(error => {
  document.querySelector("#results").textContent = "Recorded results are temporarily unavailable. The visitor guide and source code remain accessible.";
  document.querySelector("#deployment").textContent = error.message;
});
