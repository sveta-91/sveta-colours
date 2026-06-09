# Verify-script templates

Reusable scaffolds for Selenium UI verification of the portfolio site.

## Files

- `selenium_verify.py` — generic UI verify template. Loads the deployed site,
  navigates SPA-style via `showPage()`, dumps card geometry, hover-tests the
  custom cursor, screenshots the result.

## Workflow

```bash
cd tools/image_pipeline
source .venv/bin/activate
cp ../../scripts/templates/selenium_verify.py /tmp/<intent>_verify.py
# Edit the ASSERTIONS section for the specific change you're verifying.
python /tmp/<intent>_verify.py
```

One verify script per change batch, named per intent
(`/tmp/cursor_scope_verify.py`, `/tmp/landscape_span_verify.py`, etc.). The
template lives here in the repo; the per-batch copies live in `/tmp/` and
get thrown away after the batch ships.

See `docs/portfolio-playbook.md` for the full list of gotchas the template
already handles (hash routing, masonry timing, `move_to_element` reset,
modal class name).
