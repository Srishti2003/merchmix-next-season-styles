---
name: style-dna-brief
description: Turn a winning style (attributes + sales evidence + reference image) into a KEEP/CHANGE design brief and a FLUX Kontext image-edit prompt. Use when a forecast winner needs a new next-season product concept that keeps what made it sell but is clearly a new product.
---

# style-dna-brief

Input: one winning style — its `forecast.json` (outputs/evidence/<code>/), attributes, sales curve and a
reference photo. Output: one brief JSON (schema below) whose `prompt` goes straight to `image.generate_concept`.
Reusable code (template, change axes, validator, evidence digest): `skills_lib/style_dna.py`.

## Workflow

1. **Gather evidence** (do not invent numbers — every figure you cite must come from here):
   - `outputs/evidence/<code>/forecast.json`, or run `python -m skills_lib.style_dna <forecast.json>` for a digest;
   - MCP: `retail.get_style_attributes`, `retail.get_sales_curve` (units/buyers ⇒ repeat rate),
     `forecast.explain_style`;
   - **look at the reference photo** (`retail.get_reference_images` → open it with Read). Traits you KEEP must be
     visible in the photo or stated in `detail_desc`.
2. **Extract the DNA** (KEEP, 3–4 traits) — see rules.
3. **Choose the CHANGES** (2–3) — see rules.
4. **Fill the template**, write a 1–3 sentence rationale.
5. **Self-check** (list at the end; `python -m skills_lib.style_dna <forecast.json> <brief.json>` runs the automatic
   part). Fix and re-check before returning.

## 1 · DNA extraction — KEEP (3–4 traits)

- Only **visual / product** traits: silhouette, fit, length, neckline, closure, pocket/detail construction,
  fabric family. Never brand names, logos, labels or text.
- Each trait gets **evidence**: an attribute (`product_type_name`, a phrase from `detail_desc`), a sales fact
  (rank, forecast units, units per buyer, number of colourways) or a SHAP driver.
- Be honest about what the evidence says. The forecast's SHAP drivers explain **momentum** (recent units,
  discount, recency), not which visual trait sells. The strongest evidence that a trait is DNA is that the
  product **sells across many colourways** — then shape/construction is what customers buy, and colour is not.
- Prefer the traits a customer would name ("the slim ankle trouser with the elastic waist") over tiny details.

## 2 · CHANGE rules (exactly 2–3)

- Each change uses a different axis from: **silhouette detail · fabric/texture · colourway · trim/closure ·
  length/proportion · pattern**.
- **At least one must be visible at thumbnail size** (board columns are ~280 px wide): silhouette detail,
  colourway, pattern or length/proportion.
- A colourway-only change is a **recolour, not a new product** (and scores ≥ 0.80 CLIP similarity, which the
  critic rejects as `too_close`). Pair colour with a shape or pattern change.
- **Forbidden:** copying prints, graphics, text or logos (from this or any product); changing the product
  category (a trouser stays a trouser, a blouse stays a blouse); adding a model/mannequin.
- Seasonal logic helps: the forecast is for late Sep → Oct (autumn), so autumn fabrics/colours fit.
- Each change is written as `from → to`, concrete enough to draw ("brushed wool-look check in charcoal and camel",
  not "more modern").

## 3 · Output JSON schema

```json
{
  "product_code": "0751471",
  "keep":   [{"trait": "…", "evidence": "…"}],
  "change": [{"axis": "pattern", "from": "…", "to": "…"}],
  "rationale": "1–3 sentences",
  "prompt": "filled template",
  "negative_prompt": "logo, brand name, text, …"
}
```

## 4 · Prompt template (FLUX.1 Kontext image edit)

```
Redesign this garment ({product_type}) as a new next-season product. Keep: {keep traits, comma-separated}.
Change: {change "to" values, comma-separated}. Studio product photo, plain light-grey background,
same camera angle, no model, no text.
```

Kontext edits the reference photo, so name the product type (keeps the category) and describe changes as
visible outcomes. `negative_prompt` is stored for lineage; the FLUX Kontext [dev] Space has no negative-prompt
input, which is why the template itself says "no model, no text".

## 5 · Worked examples (real winners, cutoff 22 Sep 2020)

### A · Bottoms — 0751471 "Pluto RW slacks", Trousers (#1)

Evidence: #1 of 20,318 styles, 7,417 units forecast for the next 4 weeks (9,182 in the last 4); sells in 10
colours (top: Black, Dark Blue, Beige); ≈1.4 units per buyer per week over the last 4 weeks; weekly units rose
from ~700 to a 2,796 peak in early September. `detail_desc`: "Ankle-length cigarette trousers in a stretch
weave … regular waist with concealed elastication … side pockets … tapered legs."

```json
{
  "product_code": "0751471",
  "keep": [
    {"trait": "slim tapered ankle-length cigarette leg", "evidence": "detail_desc 'ankle-length cigarette … tapered legs'; sold in 10 colourways, so the shape (not the colour) is what sells"},
    {"trait": "regular waist with belt loops and concealed elastication", "evidence": "detail_desc; ≈1.4 units per buyer — a comfortable basic bought in multiples"},
    {"trait": "front slant pockets and pressed centre crease", "evidence": "visible in reference photo 0751471001.jpg"},
    {"trait": "stretch-weave tailored look", "evidence": "detail_desc 'stretch weave'; #1 forecast, 7,417 units next 4 weeks"}
  ],
  "change": [
    {"axis": "pattern", "from": "solid black", "to": "brushed wool-look check in charcoal and camel"},
    {"axis": "trim/closure", "from": "plain side seam", "to": "a narrow camel contrast side stripe"},
    {"axis": "length/proportion", "from": "closed ankle hem", "to": "cropped hem with a small side split"}
  ],
  "rationale": "Customers buy this shape in every colour, so the silhouette and comfort waist stay; an autumn check, a sporty side stripe and a split hem make it read as a new season's trouser at a glance.",
  "prompt": "Redesign this garment (trousers) as a new next-season product. Keep: slim tapered ankle-length cigarette leg, regular waist with belt loops and concealed elastication, front slant pockets and pressed centre crease, stretch-weave tailored look. Change: brushed wool-look check in charcoal and camel, a narrow camel contrast side stripe, cropped hem with a small side split. Studio product photo, plain light-grey background, same camera angle, no model, no text.",
  "negative_prompt": "logo, brand name, text, lettering, watermark, human model, mannequin, different garment category, copied print, busy background, extra garments"
}
```

### B · Top — 0762846 "Lucy blouse", Blouses (#2)

Evidence: #2 (diversified), 6,492 units forecast (6,662 in the last 4 weeks); 11 colours (top: Black, White,
Dark Green); weekly units grew from ~200 to ~1,800 over 12 weeks; ≈1.4 units per buyer. `detail_desc`:
"Long-sleeved blouse in woven fabric with a collar, V-neck, buttons down the front, buttoned cuffs and a
rounded hem."

```json
{
  "product_code": "0762846",
  "keep": [
    {"trait": "long-sleeved woven blouse with a shirt collar and open V-neck", "evidence": "detail_desc; product_type Shirt / garment group Blouses"},
    {"trait": "button-front placket and buttoned cuffs", "evidence": "detail_desc 'buttons down the front, buttoned cuffs'"},
    {"trait": "relaxed straight body with a rounded hem", "evidence": "detail_desc 'rounded hem'; visible in reference photo"},
    {"trait": "solid, single-colour base", "evidence": "graphical appearance Solid; sold in 11 colours → the shape carries the style"}
  ],
  "change": [
    {"axis": "silhouette detail", "from": "straight sleeve", "to": "voluminous balloon sleeves gathered into deep three-button cuffs"},
    {"axis": "fabric/texture", "from": "matte woven fabric", "to": "fluid satin with a soft sheen"},
    {"axis": "colourway", "from": "black / white", "to": "deep burgundy"}
  ],
  "rationale": "The collar-and-V blouse sells in every colour, so its shape stays; balloon sleeves and satin move it from office basic to autumn statement, and burgundy is on-season.",
  "prompt": "Redesign this garment (shirt) as a new next-season product. Keep: long-sleeved woven blouse with a shirt collar and open V-neck, button-front placket and buttoned cuffs, relaxed straight body with a rounded hem, solid, single-colour base. Change: voluminous balloon sleeves gathered into deep three-button cuffs, fluid satin with a soft sheen, deep burgundy. Studio product photo, plain light-grey background, same camera angle, no model, no text.",
  "negative_prompt": "logo, brand name, text, lettering, watermark, human model, mannequin, different garment category, copied print, busy background, extra garments"
}
```

## 6 · Self-check (before returning)

- [ ] 3–4 KEEP traits, each visual/product and each with evidence taken from the tools (no invented numbers).
- [ ] At least one KEEP trait is backed by the reference photo or `detail_desc`, not only by sales numbers.
- [ ] 2–3 CHANGES on different axes from the allowed list; not colourway-only.
- [ ] ≥ 1 change visible at thumbnail size (silhouette detail, colourway, pattern, length/proportion).
- [ ] No logos, brand names, text or copied prints anywhere; category unchanged; prompt names the product type.
- [ ] Prompt = the template, filled; ends with the studio-photo / no model / no text clause.
- [ ] `python -m skills_lib.style_dna <forecast.json> <brief.json>` prints `VALID`.
