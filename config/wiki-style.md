# Wiki Style Guide

Formatting conventions for all articles.

---

## Files

- Filenames are lowercase, hyphenated slugs: `topic-name.md`, `company-policy.md`
- No spaces, no uppercase, no underscores in filenames
- Keep slugs short but unambiguous: `gdpr.md` not `gdpr-regulation-overview.md`

## Sections

Standard section order:
1. Frontmatter (YAML)
2. `# Title` (H1, matches frontmatter title)
3. **Bold one-sentence lead** immediately after H1
4. `## Overview`
5. `## Key Claims`
6. `## Connections`
7. `## Sources`

Optional sections (insert before Connections if used):
- `## Architecture` — for systems and frameworks with named components
- `## History` — for topics with meaningful development timelines
- `## Criticisms` — for documented limitations or counterarguments
- `## Examples` — for abstract concepts where concrete illustration helps

## Prose Style

- Dense, not verbose. Every sentence carries information.
- Present tense for standing claims, past tense for specific events.
- No hedging filler: not "it could be argued that..." — just state the claim.
- Attribute claims: "Per [source] ([year]), ..."
- Dates always ISO format: `2024-03-15`

## Backlinks

Use typed wikilinks: `[[type:slug]]`

| Type | Use when |
|------|----------|
| `references` | General mention or citation (default for bare `[[slug]]`) |
| `depends_on` | This article's subject functionally requires the target |
| `extends` | Builds directly on or evolves from the target |
| `contradicts` | Claims in this article conflict with the target |
| `related` | Thematically connected, no directional relationship |

- Only link on first meaningful use in a section, not every occurrence
- The Connections section lists all intentional cross-references with brief rationale
- Bare `[[slug]]` is valid and treated as `references`

## Code and Technical Content

- Inline code: `backticks`
- Blocks: fenced with language tag
- Technical terms: use exact names as they appear in official sources

## Numbers and Units

- Large numbers: `1M`, `4.2B`, `32K` (no spaces before unit)
- Percentages: `42%` (no space)
- Currency: `$1,200/month`, `€45,000`
- Dates: `2024-03-15` (ISO), never "March 15th"

## What Not to Do

- No H2 headers that are just one word ("Background", "Summary", "Introduction")
- No bullet lists of 10+ items — group into subsections instead
- No placeholder text like "[TO BE FILLED]" — if you don't have the content, skip the section
- No "As an AI..." self-references in compiled content
- No invented sources — use `(source unknown)` if attribution is missing
