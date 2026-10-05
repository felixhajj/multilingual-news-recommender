"""Launch the local, model-free reviewer for the frozen 30 + 30 evaluation items."""
import argparse
import base64
import html
import json
import re
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from src.news_review import empty_labels, save_extraction_review, save_ranking_review
from src.portfolio_config import ARTIFACTS, read_json, read_jsonl

ENTITY_CATEGORIES = ("countries", "locations", "companies", "organizations", "profiles", "systems")
CATEGORY_LABELS = {
    "countries": "Countries",
    "locations": "Places (not countries)",
    "companies": "Companies",
    "organizations": "Government, military, and other organizations",
    "profiles": "People",
    "systems": "Named weapons or equipment",
}
CATEGORY_CLASSES = {
    "countries": "entity-country", "locations": "entity-location", "companies": "entity-company",
    "organizations": "entity-organization", "profiles": "entity-person",
    "systems": "entity-system",
}
REVIEW_CSS = """
.review-intro {background:#eef6f3;border-left:5px solid #176b63;padding:14px 18px;border-radius:8px;margin-bottom:12px}
.article-review {font-size:16px;line-height:1.75;background:#fff;border:1px solid #ccd7d4;border-radius:10px;padding:18px;max-height:520px;overflow:auto}
.article-review h2 {font-size:21px;margin:0 0 14px;color:#132d2a}
.article-review mark {padding:1px 4px;border-radius:4px;font-weight:650;cursor:pointer}
.entity-country {background:#d9edff}.entity-location {background:#d7efe9}.entity-company {background:#ffe5bd}.entity-organization {background:#dff2df}
.entity-person {background:#f5ddf0}.entity-system {background:#ffd9d2}
.annotation-toolbar {display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 10px}
.annotation-tool {border:1px solid transparent;border-radius:16px;padding:6px 10px;font-size:12px;cursor:pointer;color:#17302d}
.annotation-tool:hover {filter:brightness(.96);border-color:#69827e}.annotation-tool.active {outline:3px solid #173f3a;outline-offset:2px}
.annotation-remove {background:#f0efeb}.annotation-undo,.annotation-reset {background:#fff;border-color:#aab9b6}
.annotation-tool:disabled {opacity:.45;cursor:not-allowed}.annotation-status {font-size:12px;color:#516662;margin-left:3px}
.annotation-status.error {color:#a33b2f}.annotation-status.success {color:#176b63;font-weight:650}
"""

REVIEW_JS = r"""() => {
  const appRoot = () => document.querySelector('gradio-app')?.shadowRoot || document;
  const categories = ['countries', 'locations', 'companies', 'organizations', 'profiles', 'systems'];
  const classNames = {
    countries: 'entity-country', locations: 'entity-location', companies: 'entity-company', organizations: 'entity-organization',
    profiles: 'entity-person', systems: 'entity-system'
  };
  const state = {active: null, history: [], articleId: null, baseline: null, writing: false};
  const decode = value => JSON.parse(new TextDecoder().decode(
    Uint8Array.from(atob(value), character => character.charCodeAt(0))
  ));
  const escapeHtml = value => value.replace(/[&<>\"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '\"': '&quot;', "'": '&#39;'
  })[char]);
  const parse = value => [...new Set((value || '').split(/[|\n]+/).map(v => v.trim()).filter(Boolean))];
  const textbox = category => appRoot().querySelector(`#review-${category} textarea`);
  const readLists = () => Object.fromEntries(categories.map(category => [category, parse(textbox(category)?.value)]));
  const setStatus = (root, message, kind = '') => {
    const node = root.querySelector('.annotation-status');
    if (node) { node.textContent = message; node.className = `annotation-status ${kind}`; }
  };
  const renderText = (value, lists) => {
    const phraseCategory = {};
    categories.forEach(category => lists[category].forEach(phrase => { phraseCategory[phrase] = category; }));
    const phrases = Object.keys(phraseCategory).sort((a, b) => b.length - a.length);
    if (!phrases.length) return escapeHtml(value).replace(/\n/g, '<br>');
    const wordCharacter = character => Boolean(character) && /[\p{L}\p{N}_]/u.test(character);
    const matches = [];
    phrases.forEach(phrase => {
      let offset = 0;
      while (offset <= value.length - phrase.length) {
        const start = value.indexOf(phrase, offset);
        if (start < 0) break;
        const end = start + phrase.length;
        const leftOkay = !wordCharacter(phrase[0]) || start === 0 || !wordCharacter(value[start - 1]);
        const rightOkay = !wordCharacter(phrase.at(-1)) || end === value.length || !wordCharacter(value[end]);
        if (leftOkay && rightOkay) matches.push({start, end, phrase});
        offset = start + Math.max(1, phrase.length);
      }
    });
    matches.sort((a, b) => a.start - b.start || b.phrase.length - a.phrase.length);
    let output = '', cursor = 0;
    for (const match of matches) {
      if (match.start < cursor) continue;
      output += escapeHtml(value.slice(cursor, match.start));
      const phrase = match.phrase, category = phraseCategory[phrase];
      output += `<mark class="${classNames[category]}" data-phrase="${escapeHtml(phrase)}" title="Click to reclassify or remove">${escapeHtml(phrase)}</mark>`;
      cursor = match.end;
    }
    return (output + escapeHtml(value.slice(cursor))).replace(/\n/g, '<br>');
  };
  const render = root => {
    const raw = decode(root.dataset.raw);
    const lists = readLists();
    root.querySelector('.article-title').innerHTML = renderText(raw.title, lists);
    root.querySelector('.article-text').innerHTML = renderText(raw.body, lists);
  };
  const writeLists = (root, lists) => {
    state.writing = true;
    categories.forEach(category => {
      const input = textbox(category);
      if (!input) return;
      input.value = lists[category].join(' | ');
      input.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText'}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });
    state.writing = false;
    render(root);
  };
  const snapshot = () => JSON.parse(JSON.stringify(readLists()));
  const remember = root => {
    state.history.push(snapshot());
    if (state.history.length > 50) state.history.shift();
    const undo = root.querySelector('[data-action="undo"]');
    if (undo) undo.disabled = false;
  };
  const applyPhrase = (root, phrase, mode) => {
    phrase = phrase.replace(/\s+/g, ' ').trim();
    if (!phrase || phrase.length > 120) {
      setStatus(root, 'Select one exact name, no longer than 120 characters.', 'error');
      return;
    }
    const raw = decode(root.dataset.raw);
    if (!raw.title.includes(phrase) && !raw.body.includes(phrase)) {
      setStatus(root, 'That selection is not an exact phrase from the article.', 'error');
      return;
    }
    const lists = readLists();
    const exists = categories.some(category => lists[category].includes(phrase));
    if (mode === 'remove' && !exists) {
      setStatus(root, 'That phrase is not currently highlighted.', 'error');
      return;
    }
    remember(root);
    categories.forEach(category => { lists[category] = lists[category].filter(value => value !== phrase); });
    if (mode !== 'remove') lists[mode].push(phrase);
    writeLists(root, lists);
    setStatus(root, mode === 'remove' ? `Removed “${phrase}”.` : `Added “${phrase}”.`, 'success');
    window.getSelection()?.removeAllRanges();
  };
  const initialize = root => {
    if (!root || root.dataset.ready === 'true') return;
    root.dataset.ready = 'true';
    state.articleId = root.dataset.articleId;
    state.baseline = decode(root.dataset.baseline);
    state.history = [];
    state.active = null;
    root.querySelectorAll('.annotation-tool').forEach(button => button.classList.remove('active'));
    const undo = root.querySelector('[data-action="undo"]');
    if (undo) undo.disabled = true;
    setStatus(root, 'Choose a color, then select an exact name in the article.');
  };
  const currentRoot = () => appRoot().querySelector('.article-annotator');

  appRoot().addEventListener('click', event => {
    const root = event.target.closest('.article-annotator');
    if (!root) return;
    const tool = event.target.closest('.annotation-tool');
    if (tool) {
      event.preventDefault();
      const action = tool.dataset.action;
      if (action === 'undo') {
        const previous = state.history.pop();
        if (previous) writeLists(root, previous);
        tool.disabled = !state.history.length;
        setStatus(root, previous ? 'Undid the last change.' : 'Nothing to undo.', previous ? 'success' : '');
        return;
      }
      if (action === 'reset') {
        remember(root); writeLists(root, JSON.parse(JSON.stringify(state.baseline)));
        setStatus(root, 'Restored the original suggestions.', 'success');
        return;
      }
      state.active = action === 'remove' ? 'remove' : tool.dataset.category;
      root.querySelectorAll('.annotation-tool').forEach(button => button.classList.toggle('active', button === tool));
      setStatus(root, state.active === 'remove' ? 'Remove mode: select or click a highlighted name.' : `${tool.textContent.trim()} selected. Highlight an exact name.`);
      return;
    }
    const mark = event.target.closest('mark[data-phrase]');
    if (mark && state.active) applyPhrase(root, mark.dataset.phrase, state.active);
  });

  appRoot().addEventListener('mouseup', event => {
    const root = event.target.closest('.article-annotator');
    if (!root || !state.active || event.target.closest('.annotation-tool')) return;
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount) return;
    const range = selection.getRangeAt(0);
    const article = root.querySelector('.article-review');
    if (article.contains(range.commonAncestorContainer)) applyPhrase(root, selection.toString(), state.active);
  });

  appRoot().addEventListener('input', event => {
    if (state.writing || !categories.some(category => event.target === textbox(category))) return;
    const root = currentRoot();
    if (root) window.setTimeout(() => render(root), 20);
  });

  const observer = new MutationObserver(() => {
    const root = currentRoot();
    if (root) initialize(root);
  });
  observer.observe(appRoot(), {childList: true, subtree: true});
  window.setTimeout(() => initialize(currentRoot()), 100);
}"""


def candidate_map():
    value = read_json(ARTIFACTS / "review/extraction_candidates.json", {"items": []})
    return {row["article_id"]: row["candidates"] for row in value["items"]}


def parse_entity_list(value):
    return list(dict.fromkeys(part.strip() for part in re.split(r"[|\n]+", value or "") if part.strip()))


def suggested_labels(row, candidates):
    labels = {category: [] for category in ENTITY_CATEGORIES}
    if isinstance(row.get("labels"), dict):
        for category in ENTITY_CATEGORIES:
            labels[category] = list(row["labels"].get(category, []))
    if row["review_status"] == "human_reviewed":
        return labels
    for candidate in candidates:
        category = candidate["suggested_category"]
        if category in labels and candidate["mention"] not in labels[category]:
            labels[category].append(candidate["mention"])
    return labels


def highlighted_article(row, labels):
    categories = {mention: category for category in ENTITY_CATEGORIES for mention in labels[category]}

    def phrase_pattern(value):
        result = re.escape(value)
        if value[0].isalnum() or value[0] == "_":
            result = rf"(?<!\w){result}"
        if value[-1].isalnum() or value[-1] == "_":
            result = rf"{result}(?!\w)"
        return result

    pattern = re.compile("|".join(phrase_pattern(value) for value in sorted(categories, key=len, reverse=True))) if categories else None

    def render(value):
        if not pattern:
            return html.escape(value).replace("\n", "<br>")
        output, cursor = [], 0
        for match in pattern.finditer(value):
            output.append(html.escape(value[cursor:match.start()]))
            phrase = match.group(0)
            category = categories[phrase]
            output.append(
                f'<mark class="{CATEGORY_CLASSES[category]}" title="{html.escape(CATEGORY_LABELS[category])}">'
                f'{html.escape(phrase)}</mark>'
            )
            cursor = match.end()
        output.append(html.escape(value[cursor:]))
        return "".join(output).replace("\n", "<br>")

    raw = base64.b64encode(json.dumps({"title": row["title"], "body": row["text"]}, ensure_ascii=False).encode()).decode()
    baseline = base64.b64encode(json.dumps(labels, ensure_ascii=False).encode()).decode()
    tools = "".join(
        f'<button type="button" class="annotation-tool {CATEGORY_CLASSES[key]}" data-category="{key}">{label}</button>'
        for key, label in CATEGORY_LABELS.items()
    )
    return (
        f'<section class="article-annotator" data-article-id="{html.escape(row["article_id"])}" '
        f'data-raw="{raw}" data-baseline="{baseline}"><div class="annotation-toolbar">{tools}'
        '<button type="button" class="annotation-tool annotation-remove" data-action="remove">Remove highlight</button>'
        '<button type="button" class="annotation-tool annotation-undo" data-action="undo">Undo</button>'
        '<button type="button" class="annotation-tool annotation-reset" data-action="reset">Reset suggestions</button>'
        '<span class="annotation-status"></span></div><article class="article-review" dir="auto">'
        f'<h2 class="article-title">{render(row["title"])}</h2>'
        f'<div class="article-text">{render(row["text"])}</div></article></section>'
    )


def extraction_item(index):
    rows = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    index = max(0, min(int(index), len(rows) - 1))
    row = rows[index]
    candidates = candidate_map().get(row["article_id"], [])
    labels = suggested_labels(row, candidates)
    attribution = f" by {row['reviewed_by']}" if row.get("reviewed_by") else ""
    heading = f"Item {index + 1}/30 | {row['language'].upper()} | {row['article_id']} | {row['review_status']}{attribution}"
    return (index, heading, highlighted_article(row, labels), row["source_url"],
            *(" | ".join(labels[category]) for category in ENTITY_CATEGORIES), False)


def ranking_item(index):
    rows = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
    index = max(0, min(int(index), len(rows) - 1))
    row = rows[index]
    article = row["article"]
    attribution = f" by {row['reviewed_by']}" if row.get("reviewed_by") else ""
    heading = f"Item {index + 1}/30 | {row['query_id']} | {article['article_id']} | {row['review_status']}{attribution}"
    filters = json.dumps(row["required_filters"], ensure_ascii=False)
    grade = str(row["relevance"]) if row["review_status"] == "human_reviewed" else None
    return index, heading, row["interest"], filters, article["title"], article["body"], article["source_url"], grade


def save_extraction(index, countries, locations, companies, organizations, profiles, systems, checked, reviewer):
    if not checked:
        raise gr.Error("Confirm that you checked the article for missing named entities")
    rows = read_jsonl(ARTIFACTS / "review/extraction.jsonl")
    row = rows[int(index)]
    labels = empty_labels()
    for category, text in zip(ENTITY_CATEGORIES, (countries, locations, companies, organizations, profiles, systems)):
        labels[category] = parse_entity_list(text)
    count = save_extraction_review(row["article_id"], labels, reviewer, scope="named_entities_v3")
    return f"Saved human review. Extraction progress: {count}/30", *extraction_item(min(int(index) + 1, 29))


def save_ranking(index, grade, reviewer):
    rows = read_jsonl(ARTIFACTS / "review/ranking.jsonl")
    row = rows[int(index)]
    if grade is None:
        raise gr.Error("Choose 0, 1 or 2")
    count = save_ranking_review(row["query_id"], row["article"]["article_id"], int(grade), reviewer)
    return f"Saved human judgment. Ranking progress: {count}/30", *ranking_item(min(int(index) + 1, 29))


def app():
    with gr.Blocks(title="Frozen evaluation review", css=REVIEW_CSS, js=REVIEW_JS) as demo:
        gr.Markdown("# Human review\nYou verify independent candidates; you do not search for relationships or edit JSON. The screen never shows predictions from the models being evaluated.")
        reviewer = gr.Textbox(label="Reviewer name or stable ID", placeholder="Required before saving")
        with gr.Tab("Extraction: 30 articles"):
            gr.HTML("""<div class="review-intro"><strong>Your job is verification, not extraction from scratch.</strong><br>
            Choose a color and select a name directly in the article. The matching list updates automatically.
            Use Remove, Undo, or Reset when needed. Keep the exact article spelling.</div>""")
            ex_index = gr.Slider(0, 29, value=0, step=1, label="Item")
            ex_heading = gr.Markdown()
            ex_body = gr.HTML()
            ex_source = gr.Textbox(label="Source", interactive=False)
            gr.Markdown("### Verify these prefilled lists\nCountries are sovereign states. Places are named non-country geography such as Gaza, cities, regions, seas, straits, borders, and named bases. Delete wrong entries or move them between boxes.")
            with gr.Row():
                extra_countries = gr.Textbox(label="Countries", lines=3, placeholder="Iran | Lebanon",
                                             elem_id="review-countries")
                extra_locations = gr.Textbox(label="Places (not countries)", lines=3,
                                             placeholder="Gaza Strip | Beirut | Red Sea",
                                             elem_id="review-locations")
            with gr.Row():
                extra_companies = gr.Textbox(label="Companies", lines=3, placeholder="Lockheed Martin",
                                             elem_id="review-companies")
            with gr.Row():
                extra_organizations = gr.Textbox(label="Government, military, and other organizations", lines=3,
                                                 placeholder="United Nations | Ministry of Defence",
                                                 elem_id="review-organizations")
                extra_profiles = gr.Textbox(label="People", lines=3, placeholder="Mohammed bin Salman",
                                            elem_id="review-profiles")
            extra_systems = gr.Textbox(label="Named weapons or equipment", lines=2,
                                       placeholder="MQ-9B | Patriot (not generic words such as missile)",
                                       elem_id="review-systems")
            gr.Markdown("Do **not** add topics, relationships, dates, job titles, nationality adjectives, unnamed places, or generic equipment.")
            checked = gr.Checkbox(label="I verified the highlighted article and the six lists")
            ex_save = gr.Button("Save and open next article", variant="primary")
            ex_status = gr.Markdown()
            ex_outputs = [ex_index, ex_heading, ex_body, ex_source, extra_countries, extra_locations, extra_companies,
                          extra_organizations, extra_profiles, extra_systems, checked]
            ex_index.change(extraction_item, ex_index, ex_outputs)
            ex_save.click(save_extraction, [ex_index, extra_countries, extra_locations, extra_companies, extra_organizations,
                          extra_profiles, extra_systems, checked, reviewer], [ex_status, *ex_outputs])
            demo.load(extraction_item, gr.State(0), ex_outputs)
        with gr.Tab("Recommendation: 30 judgments"):
            rank_index = gr.Slider(0, 29, value=0, step=1, label="Item")
            rank_heading = gr.Markdown()
            interest = gr.Textbox(label="User interest", interactive=False)
            required = gr.Textbox(label="Required filters", interactive=False)
            rank_title = gr.Textbox(label="Article title", interactive=False)
            rank_body = gr.Textbox(label="Article", lines=18, interactive=False)
            rank_source = gr.Textbox(label="Source", interactive=False)
            grade = gr.Radio(["0", "1", "2"], label="Relevance: 0 irrelevant, 1 partial, 2 direct")
            rank_save = gr.Button("Save recommendation judgment", variant="primary")
            rank_status = gr.Markdown()
            rank_outputs = [rank_index, rank_heading, interest, required, rank_title, rank_body, rank_source, grade]
            rank_index.change(ranking_item, rank_index, rank_outputs)
            rank_save.click(save_ranking, [rank_index, grade, reviewer], [rank_status, *rank_outputs])
            demo.load(ranking_item, gr.State(0), rank_outputs)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio public tunnel")
    parser.add_argument("--username", default="reviewer")
    parser.add_argument("--password", help="Required for remote review; generated if --share is used")
    args = parser.parse_args()
    password = args.password or (secrets.token_urlsafe(12) if args.share else None)
    if args.share:
        print(f"REVIEW_USERNAME={args.username}", flush=True)
        print(f"REVIEW_PASSWORD={password}", flush=True)
    demo = app()
    _, local_url, share_url = demo.launch(
        server_name=args.host, server_port=args.port, inbrowser=not args.share,
        share=args.share, auth=(args.username, password) if password else None,
        prevent_thread_lock=True,
    )
    print(f"REVIEW_LOCAL_URL={local_url}", flush=True)
    print(f"REVIEW_SHARE_URL={share_url or ''}", flush=True)
    demo.block_thread()
