"""Render the visitor walkthrough from frozen results and recorded model outputs.

Requires reportlab. This reads artifacts only; it never loads or trains models.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://felixhajj.github.io/multilingual-news-recommender/"
CODE = "https://github.com/felixhajj/multilingual-news-recommender/blob/main/1-tactical-recommendation-prototype/"
INK, MUTED, TEAL = "#163b41", "#50696a", "#08766e"
PAPER, WASH, LINE = "#f6f5ee", "#e5eee8", "#c8d7d1"
BLUE, BLUE_WASH, RUST = "#315c75", "#e8eff3", "#a44725"
WIDTH, HEIGHT, LEFT = 595.28, 841.89, 46
CONTENT = WIDTH - LEFT * 2


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Walkthrough:
    def __init__(self, output, fonts=None):
        self.body, self.bold, self.title = "Helvetica", "Helvetica-Bold", "Times-Roman"
        if fonts and (fonts / "trebuc.ttf").exists() and (fonts / "georgia.ttf").exists():
            for name, file in [("Body", "trebuc.ttf"), ("BodyBold", "trebucbd.ttf"), ("Display", "georgia.ttf")]:
                pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
            self.body, self.bold, self.title = "Body", "BodyBold", "Display"
        self.c = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), invariant=1)
        self.c.setTitle("Multilingual News Recommender Technical Walkthrough")
        self.c.setAuthor("Felix Hajj")
        self.page = 0

    def text(self, text, x, top, size=11, font=None, color=INK):
        self.c.setFillColor(colors.HexColor(color))
        self.c.setFont(font or self.body, size)
        self.c.drawString(x, HEIGHT - top - size, text)

    def para(self, text, x, top, width, size=11, color=INK, font=None):
        style = ParagraphStyle("copy", fontName=font or self.body, fontSize=size,
                               leading=size * 1.42, textColor=colors.HexColor(color), alignment=TA_LEFT)
        p = Paragraph(text, style)
        _, height = p.wrap(width, HEIGHT)
        if top + height > HEIGHT - 46:
            raise ValueError(f"Text exceeds page {self.page}: {text[:65]}")
        p.drawOn(self.c, x, HEIGHT - top - height)
        return top + height

    def rect(self, x, top, width, height, fill=WASH, stroke=None):
        self.c.setFillColor(colors.HexColor(fill))
        if stroke:
            self.c.setStrokeColor(colors.HexColor(stroke))
        self.c.roundRect(x, HEIGHT - top - height, width, height, 4, fill=1, stroke=bool(stroke))

    def line(self, x1, top1, x2, top2, color=LINE, width=1):
        self.c.setStrokeColor(colors.HexColor(color)); self.c.setLineWidth(width)
        self.c.line(x1, HEIGHT-top1, x2, HEIGHT-top2)

    def arrow(self, x1, top1, x2, top2, color=TEAL):
        self.line(x1, top1, x2, top2, color, 1.2)
        if top1 == top2:
            tail = -5 if x2 >= x1 else 5
            self.line(x2+tail, top2-3, x2, top2, color); self.line(x2+tail, top2+3, x2, top2, color)
        else:
            self.line(x2-3, top2-5, x2, top2, color); self.line(x2+3, top2-5, x2, top2, color)

    def box(self, x, top, width, height, label, title, copy, fill=WASH, color=TEAL):
        self.rect(x, top, width, height, fill)
        self.text(label.upper(), x+14, top+12, 8.5, self.bold, color)
        title_bottom = self.para(escape(title), x+14, top+30, width-28, 15, INK, self.title)
        if copy:
            bottom = self.para(escape(copy), x+14, max(top+66, title_bottom+7), width-28, 10, MUTED)
            if bottom > top+height-5:
                raise ValueError(f"Card overflows on page {self.page}: {title}")

    def new(self, label, title, intro):
        if self.page:
            self.c.showPage()
        self.page += 1
        self.c.setFillColor(colors.HexColor(PAPER)); self.c.rect(0,0,WIDTH,HEIGHT,fill=1,stroke=0)
        self.text("BETWEEN LANGUAGES", LEFT, 27, 9, self.bold, TEAL)
        self.text("FELIX HAJJ / TECHNICAL WALKTHROUGH", 298, 27, 8, self.body, MUTED)
        self.text(f"{self.page:02d} / {label.upper()}", LEFT, 68, 9, self.bold, TEAL)
        end = self.para(title, LEFT, 89, CONTENT, 28, INK, self.title)
        self.para(intro, LEFT, end+12, CONTENT, 11.2, MUTED)
        self.line(LEFT, HEIGHT-40, WIDTH-LEFT, HEIGHT-40)
        self.text("English + Arabic / research release / October 2026", LEFT, HEIGHT-29, 8, color=MUTED)
        self.text(f"{self.page} / 6", WIDTH-LEFT-24, HEIGHT-29, 8, color=MUTED)
        self.c.bookmarkPage(f"page-{self.page}"); self.c.addOutlineEntry(title, f"page-{self.page}", level=0)

    def link(self, label, url, x, top, size=9.5):
        self.text(label, x, top, size, color=TEAL)
        width = pdfmetrics.stringWidth(label, self.body, size)
        self.c.linkURL(url, (x, HEIGHT-top-size-3, x+width, HEIGHT-top+2), relative=0)

    def table(self, top, columns, rows, widths, row_height=32):
        self.rect(LEFT, top, CONTENT, row_height, WASH)
        x=LEFT
        for name, width in zip(columns, widths):
            self.para(escape(name), x+10, top+8, width-20, 9, INK, self.bold); x += width
        for i, row in enumerate(rows):
            y=top+(i+1)*row_height; x=LEFT
            for value, width in zip(row, widths):
                self.para(escape(str(value)), x+10, y+8, width-20, 10); x += width
            self.line(LEFT,y+row_height,WIDTH-LEFT,y+row_height)
        return top+(len(rows)+1)*row_height


def main(args):
    artifacts = args.artifacts
    evidence = read(ROOT / "data/release/evidence.json")
    ranking = read(ROOT / "data/release/example_ranking.json")
    manifest = read(artifacts / "runs/extraction-only-v3/manifest.json")
    losses = [json.loads(x) for x in (artifacts / "runs/extraction-only-v3/loss.jsonl").read_text().splitlines() if x]
    validations = {name: read(artifacts / f"phase3/reports/{name}_validation.json")["metrics"]
                   for name in ["base-v3", "extraction-only-v3", "domain-extraction-v3"]}
    row = ranking["results"][0]
    with sqlite3.connect((artifacts / "release.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as db:
        actual = json.loads(db.execute("select payload from analyses where cache_key=?", (row["cache_key"],)).fetchone()[0])
    link = next(x for x in actual["entity_links"] if x["mention"] == "George W. Bush")
    unresolved = next(x for x in actual["entity_links"] if x["mention"] == "Donald Rumsfeld")
    metrics = evidence["extraction_evaluation"]["metrics"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = Walkthrough(args.output, args.fonts)

    w.new("Purpose and overview", "Multilingual news recommendation",
          "Write an interest, discover relevant historical news, then inspect the facts and meaning signals behind the recommendation.")
    w.text("ONE ARTICLE, TWO PATHS", LEFT, 204, 9, w.bold, TEAL)
    w.box(LEFT, 228, 155, 118, "Article facts", "Qwen + QLoRA", "Extract structured mentions from article text.")
    w.arrow(204,287,220,287)
    w.box(225,228,155,118,"Context + name","Learned linker","Choose an entity ID or leave the mention unresolved.")
    w.arrow(384,287,400,287)
    w.box(405,228,144,118,"Optional","Filter matches","Compare article facts with requested filters.")
    w.box(LEFT,366,170,118,"Natural text","Article + interest","Both texts independently go to E5.",BLUE_WASH,BLUE)
    w.arrow(220,425,241,425,BLUE)
    w.box(246,366,303,118,"Semantic retrieval","Multilingual E5","Produce 768-dimensional vectors and compare meaning across languages.",BLUE_WASH,BLUE)
    # Route filter matches around E5 so the two input paths stay independent.
    w.line(549,287,564,287,TEAL);w.line(564,287,564,536,TEAL);w.arrow(564,536,549,536,TEAL)
    w.arrow(397,486,397,501,BLUE)
    w.rect(LEFT,502,CONTENT,69,INK)
    w.text("A ranked reading list",LEFT+17,514,19,w.title,"#ffffff")
    w.para("Open a result to inspect extracted names, links, filter matches and semantic probes.",LEFT+17,542,CONTENT-34,10,"#d6e5e1")
    w.text("DEVELOPMENT SEQUENCE / AUGUST TO OCTOBER 2026",LEFT,603,9,w.bold,TEAL)
    stages=[("Prototype","Expose each model stage"),("Real data","Collect, label and review"),("Validation","Choose the extraction adapter"),("Release","Index 300 articles and host")]
    for i,(title,copy) in enumerate(stages):
        x=LEFT+i*128
        w.text(f"{i+1:02d}",x,628,17,w.title,TEAL);w.text(title,x,655,10,w.bold)
        w.para(copy,x,674,112,9,MUTED)
    w.link("Try the live demo", "https://huggingface.co/spaces/felixhajj/multilingual-news-recommender", LEFT,744)
    w.link("Read the code",CODE,LEFT+190,744)

    w.new("Data preparation", "Real articles and traceable labels",
          "Source attribution and split discipline make the training and evaluation interpretable. Collection, training and release sizes serve different purposes.")
    w.box(LEFT,216,CONTENT,81,"Collection","5,000 historical Wikinews articles","")
    w.text("3,539 English / 1,461 Arabic",LEFT+17,268,11,color=MUTED)
    w.arrow(298,302,298,325)
    w.box(LEFT,331,CONTENT,108,"Preparation","Clean and group before splitting","Remove boilerplate and duplicates; group related versions. Retain source URL, revision, contributor attribution, license and content hash.")
    w.arrow(298,444,298,466)
    w.box(LEFT,472,244,111,"Training input","30 accepted examples","Machine-assisted article-to-JSON labels accepted from 160 attempts.")
    w.box(LEFT+259,472,244,111,"Human evaluation","60 completed review items","30 extraction references plus 30 recommendation judgments.",BLUE_WASH,BLUE)
    w.para("Published hyperlinks provide incomplete mention annotations. They cannot certify that an article has no other entities. Machine-assisted empty lists remain weak judgments.",LEFT,607,CONTENT,11)
    w.para("The archive uses 2019 snapshots referenced by Mewsli-9. This custom reconstruction is not the official Mewsli benchmark. Translation grouping is heuristic; residual overlap is possible.",LEFT,665,CONTENT,10.5,MUTED)
    w.link("Source and license details", SITE+"model-card.html", LEFT,743)
    w.link("Collection code",CODE+"src/news_corpus.py",LEFT+220,743)

    w.new("Extraction and training", "Teach Qwen to produce structured facts",
          "Qwen2.5-3B supplies language knowledge. QLoRA freezes the base in 4-bit memory and trains small LoRA updates using article/answer examples. The tokenizer vocabulary stays unchanged.")
    w.text("Text > token IDs > Qwen + trained adapter > extraction JSON",LEFT,207,10,w.bold,TEAL)
    w.rect(LEFT,236,CONTENT,224,"#ffffff",LINE)
    w.text("ACTUAL TRAINING LOSS / EXTRACTION-ONLY-V3",LEFT+16,250,9,w.bold,TEAL)
    x0,y0,gw,gh=LEFT+48,423,CONTENT-79,121
    maximum=max(x["loss"] for x in losses)*1.1
    for tick in [0,maximum/2,maximum]:
        y=y0-gh*tick/maximum;w.line(x0,y,x0+gw,y)
        w.text(f"{tick:.2f}",LEFT+13,y-5,8,color=MUTED)
    for a,b in zip(losses,losses[1:]):
        w.line(x0+gw*(a["step"]-1)/47,y0-gh*a["loss"]/maximum,x0+gw*(b["step"]-1)/47,y0-gh*b["loss"]/maximum,TEAL,1.4)
    w.text("1",x0,432,8,color=MUTED);w.text("48 optimizer steps",x0+gw-100,432,8,color=MUTED)
    w.para(f"{manifest['unique_articles_seen']} examples seen; {manifest['changed_tensors']} adapter tensors changed. Loss is the mean of four supervised-token microbatch means per update. Different batches vary; lower training loss alone does not establish accuracy.",LEFT,477,CONTENT,10,MUTED)
    rows=[]
    for name,label in [("base-v3","Base Qwen"),("extraction-only-v3","Extraction only (selected)"),("domain-extraction-v3","Domain + extraction")]:
        m=validations[name];rows.append((label,f"{m['micro_f1']:.3f}",f"{m['schema_validity']:.0%}"))
    w.table(555,["10 validation articles","Entity F1","Schema valid"],rows,[285,108,110],29)
    w.para("Selection required schema validity >= max(80%, base), improved overall F1 and no language F1 regression. Domain adaptation was evaluated but not selected. Earlier interrupted and unsuccessful runs remain in the evidence history.",LEFT,684,CONTENT,10,MUTED)
    w.link("Training notebook", CODE+"notebooks/tactical_report_entity_extraction_qlora.ipynb",LEFT,750)
    w.link("Shared trainer",CODE+"src/news_training_v3.py",LEFT+205,750)

    w.new("Entity linking", "Connect mentions using context",
          "A generated name is not an entity ID. E5 retrieves candidates from a limited Wikidata catalogue; a supervised logistic classifier ranks them using contextual and name features.")
    w.text("ACTUAL RECORDED ARTICLE / EN-39534",LEFT,207,9,w.bold,TEAL)
    w.rect(LEFT,230,CONTENT,77)
    w.para('Article mentions "U.S. President George W. Bush" during a report about Iran nuclear diplomacy. Qwen extracted "George W. Bush" as a person.',LEFT+17,246,CONTENT-34,11)
    w.arrow(297,311,297,333)
    candidates=link["candidates"]
    w.table(340,["Retrieved candidate","Learned score"],[(x["name"],f"{x['score']:.4f}") for x in candidates],[360,143],32)
    w.box(LEFT,490,244,113,"Accepted",f"{link['canonical_name']} / {link['entity_id']}","The top candidate passes the acceptance rule.")
    w.box(LEFT+259,490,244,113,"Unresolved",unresolved["mention"],"Low candidate scores leave this mention without an ID.",BLUE_WASH,BLUE)
    overall=evidence["linker"]["overall"]
    w.para(f"On the previously reported article-disjoint holdout, reused for this release ({overall['mentions']} mentions): {overall['accepted_precision']:.1%} precision among accepted links, with {overall['coverage']:.1%} coverage ({overall['accepted']} accepted). Abstention reduces false links but limits coverage.",LEFT,624,CONTENT,11)
    w.para("Scores are not calibrated probabilities. The catalogue covers a subset of Wikidata; unseen and ambiguous aliases remain difficult. Source text supports a mention, but does not prove every generated relationship.",LEFT,691,CONTENT,10,MUTED)
    w.link("Original article and contributors",row["article"]["source_url"],LEFT,752)
    w.link("Linker implementation",CODE+"src/learned_linker.py",LEFT+245,752)

    w.new("Recommendation", "Compare meaning, then add constraints",
          "E5 compares the user's written interest with natural article text. It uses pretrained multilingual knowledge directly, so Arabic and English can share a meaning space without manual translation.")
    w.box(LEFT,219,CONTENT,82,"Query input",ranking["interest"],"")
    w.box(LEFT,322,244,129,"Article input","Historical article text","Title and body use the passage: prefix. Article vectors are computed once.",BLUE_WASH,BLUE)
    w.box(LEFT+259,322,244,129,"Interest input","Written interest","The interest uses the query: prefix. A new interest gets a fresh vector.",BLUE_WASH,BLUE)
    w.arrow(166,455,166,475,BLUE);w.arrow(425,455,425,475,BLUE)
    w.rect(LEFT,483,CONTENT,78,INK)
    w.text("768 values per normalized vector",LEFT+17,497,17,w.title,"#ffffff")
    w.para("Dot product = cosine similarity. E5 receives natural text, independently of the extraction JSON.",LEFT+17,527,CONTENT-34,10,"#d6e5e1")
    w.para("Without filters, rank by E5 similarity. With filters, complete matches rank first; within each group, sort by 75% E5 similarity + 25% exact filter coverage. Both signals remain visible in the app.",LEFT,584,CONTENT,11)
    w.para(f"Recorded first result: {escape(row['article']['title'])}. Similarity/ranking score {row['score']:.3f} for this interest. The number orders results; it is not a confidence percentage.",LEFT,660,CONTENT,10,MUTED)
    w.para("Dynamic phrase probes score smaller text pairs to illustrate related passages. They do not identify the internal reason for a whole-article score.",LEFT,719,CONTENT,9.5,MUTED)
    w.link("Ranking and shared pipeline",CODE+"src/news_pipeline.py",LEFT,760)
    w.link("Recommendation notebook",CODE+"notebooks/tactical_report_recommendation_demo.ipynb",LEFT+240,760)

    w.new("Results and reproduction", "Measured performance and next steps",
          "The release works end to end, with limited model quality. Frozen references and saved predictions make the reported measurements reproducible.")
    w.text("EXTRACTION / 20 HUMAN-REVIEWED TEST ARTICLES",LEFT,202,9,w.bold,TEAL)
    w.table(224,["Parsing valid","Schema valid","Precision","Recall","Entity F1"],
            [(f"{metrics['json_parse_validity']:.0%}",f"{metrics['schema_validity']:.0%}",f"{metrics['precision']:.1%}",f"{metrics['recall']:.1%}",f"{metrics['micro_f1']:.3f}")],[105,105,99,99,95],32)
    w.para("Ten English and ten Arabic articles, held out from our training. Entity F1 uses normalized surface names; topics and relationships are schema-checked but not human-scored.",LEFT,302,CONTENT,9.5,MUTED)
    w.text("RETRIEVAL / TWO TEST QUERIES, TEN JUDGED ARTICLES EACH",LEFT,355,9,w.bold,TEAL)
    rt=evidence["recommendation_evaluation"]["test_averages"]
    w.table(379,["Method","nDCG@5","Recall@5"],[(label,f"{rt[key]['ndcg_at_5']:.3f}",f"{rt[key]['recall_at_5']:.3f}") for key,label in [("keyword","Keyword"),("e5","E5"),("hybrid","Hybrid")]],[285,108,110],29)
    w.para("Keyword and E5 rank all ten candidates per pool; hybrid ranks only eight and nine after extraction failures. These tiny pools do not establish full-corpus superiority. E5 and hybrid have equal reported averages.",LEFT,507,CONTENT,9.5,MUTED)
    w.rect(LEFT,563,CONTENT,96,"#efe5dd")
    w.text("A FAILURE TO INSPECT",LEFT+16,576,9,w.bold,RUST)
    w.para("The hosted model classified Beirut as a country. A European energy/gas query also returned an unrelated Google-slander article near the top. Failures remain visible alongside successful examples.",LEFT+16,599,CONTENT-32,10)
    w.para("Frozen quality metrics use local 4-bit Qwen; hosted inference uses the same adapter with float16 base weights. Quality parity has not been measured. Free GPU quotas may interrupt inference.",LEFT,674,CONTENT,9.5,MUTED)
    w.link("Reproduce saved metrics and open notebooks",SITE+"guide.html",LEFT,731)
    w.link("Results and limitations",SITE+"results.html",LEFT,754)
    w.link("Data, licenses and model versions",SITE+"model-card.html",LEFT+235,754)
    w.c.save()

    receipt={"pages":6,"output_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),
             "evidence_sha256":hashlib.sha256((ROOT/'data/release/evidence.json').read_bytes()).hexdigest(),
             "training_steps":manifest['optimizer_steps'],"loss_points":len(losses),
             "recorded_example_cache_key":row['cache_key'],"example_entity_id":link['entity_id']}
    args.output.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=ROOT/'output/portfolio')
    parser.add_argument('--output',type=Path,default=ROOT.parent/'docs/walkthrough.pdf')
    parser.add_argument('--fonts',type=Path,default=Path('C:/Windows/Fonts'))
    main(parser.parse_args())
