# Repository Guidelines

## Project Structure & Module Organization

This repository is a static GitHub Pages personal site. The main page is
`index.html`, which contains the HTML, embedded CSS, Google Analytics snippet,
and page content. Static files live under `assets/`: `assets/papers/` contains
publication thumbnails, `assets/Daiwei_Chen.jpg` is the profile image, and
`assets/DaiweiChen_CV.pdf` is the downloadable CV. There is no separate build
directory, package manifest, or test tree at present.

## Build, Test, and Development Commands

No package manager or build step is required. Useful local commands:

```powershell
Start-Process .\index.html
```

Opens the site in the default browser for a quick visual check.

```powershell
python -m http.server 8000
```

Serves the repository locally at `http://localhost:8000/`; use this when
checking browser behavior that depends on HTTP instead of direct file loading.

```powershell
git status --short
```

Reviews pending changes before committing.

## Coding Style & Naming Conventions

Keep the site dependency-light and edit `index.html` directly unless a larger
refactor is needed. Follow the existing four-space indentation in HTML, CSS, and
inline JavaScript. Use semantic section IDs and class names such as
`publications`, `side-nav`, or `paper-card`. Keep asset filenames descriptive
and stable, especially for files linked from the page. Prefer relative paths
like `assets/papers/star.png`.

## Useful Notes Markdown Workflow

Useful Notes live under `Good_Notes/` and load markdown from `assets/md_src/`.
When incorporating an external markdown note, copy it into `assets/md_src/` with
a lowercase, hyphenated filename, then add one entry to
`assets/md_src/notes.js`. Each entry must include `slug`, `title`, `summary`,
`date` in `MM/DD/YYYY` format, `tag`, and a relative `src` such as
`assets/md_src/my-note.md`.

Keep note titles and summaries in `notes.js`; the markdown file should contain
the note body. The renderer in `Good_Notes/note.html` already strips YAML
frontmatter and duplicate top-level `# Title` headings. Preserve Obsidian
callouts like `> [!NOTE]` and `> [!IMPORTANT]`, markdown tables, ordered lists,
and LaTeX math using `$...$` or `$$...$$`; the page renders these through its
local parser plus MathJax. Do not convert these structures manually to HTML
unless the renderer cannot support a required construct.

## Testing Guidelines

There is no automated test suite. Before submitting changes, open the page in a
browser and verify layout, navigation anchors, external links, images, and the
CV link. For publication updates, confirm that each thumbnail exists under
`assets/papers/` and that titles, venues, and links match the intended content.
Check responsive behavior with browser dev tools for both mobile and desktop
widths.

For Useful Notes, test through HTTP rather than `file:///`, because markdown is
loaded with `fetch()`. Use `npx serve . -l 8000`, then open
`http://localhost:8000/Good_Notes/` and the generated note URL.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or descriptive messages such as `update`,
`add conformal paper and reorder publications`, and `move LinkedOut to selected
publications...`. Prefer a specific summary over a generic one when possible,
for example `add new publication thumbnail` or `update CV link`.

Pull requests should describe the visible site change, list affected assets or
sections, and include screenshots for layout or styling changes. Link related
issues when applicable and note any manual browser checks performed.

## Security & Configuration Tips

Do not commit local OS metadata; `.gitignore` already excludes common macOS
files. Treat analytics IDs and external links as production-facing content, and
verify them before changing.
