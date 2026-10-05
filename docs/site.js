const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
async function loadRanking() {
  const response = await fetch("example_ranking.json");
  if (!response.ok) throw new Error("Recorded ranking unavailable");
  const ranking = await response.json();
  if (!ranking.results?.length) return;
  const cards = [];
  for (const [index, row] of ranking.results.slice(0, 3).entries()) {
    const article = row.article;
    const card = element("article", undefined, "record");
    card.append(element("span", String(index + 1).padStart(2, "0"), "position"));
    const content = element("div");
    content.append(element("small", article.language === "ar" ? "ARABIC" : "ENGLISH", "article-language"));
    const heading = element("h3", article.title); heading.dir = "auto"; content.append(heading);
    const excerpt = element("p", article.body.slice(0, 210) + (article.body.length > 210 ? "..." : "")); excerpt.dir = "auto"; content.append(excerpt);
    if (/^https:\/\//.test(article.source_url || "")) { const link = element("a", "Read the original article"); link.href = article.source_url; content.append(link); }
    const dateLabel = article.date_kind === "revision_date_publication_unknown" ? "Archived revision" : "Source date";
    content.append(element("p", `${article.author || "Source contributors"} / ${article.license_id || "See source"}${article.date ? ` / ${dateLabel}: ${article.date}` : ""}`, "small attribution"));
    const details = element("details", undefined, "score-detail"); details.append(element("summary", "Inspect the recorded score"));
    details.append(element("p", `Ranking score: ${row.score.toFixed(3)}. ${row.score_note || "Scores indicate relative relevance, not confidence percentages."}`, "small"));
    content.append(details); card.append(content); cards.push(card);
  }
  document.querySelector("#interest").textContent = ranking.interest;
  document.querySelector("#results").replaceChildren(...cards);
}
async function loadCounts() {
  const response = await fetch("evidence.json"); if (!response.ok) return;
  const evidence = await response.json();
  for (const [id, value] of [["corpus-count", evidence.corpus?.documents], ["release-count", evidence.release?.articles]]) {
    if (Number.isInteger(value)) document.getElementById(id).textContent = value.toLocaleString("en-US");
  }
}
// Static text and links remain usable when an optional data request fails.
loadRanking().catch(() => {});
loadCounts().catch(() => {});
