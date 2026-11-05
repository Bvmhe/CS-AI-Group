# Canva upload: where we stopped (2026-11-04)

- Design: DAHZ9YV1UKE (it still holds only the old landscape page 1, PBGwMgfLWRRc9Tp5).
- Pages rendered: 257 portrait pages (1080×1920). Test Bank docx done (58 questions).
- Uploaded to Canva: pages 1–114. The mediaIds are in `media.json`. 143 pages are left (115–257).

## Update 2026-11-05
- The cover title is now "Fengwei Fried Rice", and the front-matter chapter label too.
- Chapter 1 example pictures were added with the new `PIC()` block: a new page 1.4p (memory pyramid), a new page 1.5p (A17 Pro), and pictures on 1.5b and 1.5c. The guide now has 259 pages.
- Waiting for the other session to add example pictures for Chapters 2–5 (Main/Chapter-0X/assets). Then add a PIC() for each and re-render.
- Re-upload rule: `uploaded_md5.json` holds the hash of each uploaded page image (by page number). Re-upload any page whose new render hash differs from it.

## To continue
1. Re-render the pages (the temp folder may be gone):
   `cd Study-Guide/tools; python render_pages.py <out>`
2. Request create-upload-url in bursts of about 8–12, waiting about 60 s between bursts (rate limit).
   Then run: `$env:PAGES_DIR="<out>\full"; python -I Study-Guide/tools/canva/up2.py`
3. When all 257 pages have mediaIds, build the pages:
   - In one transaction on DAHZ9YV1UKE: add_page ×257 (1080×1920).
   - Read the page ids with design_content and page_indices.
   - insert_fill each page with its image (0, 0, 1080, 1920).
4. Show a preview and get the user's approval, then commit the transaction. Delete old page 1 only with approval.
5. Export the PDF to Study-Guide/exports/.
6. Write the AI-Logs entry.
7. Build the ZIP `FengweiFriedRice_Stage2_StudyGuide_TestBank_V1.zip` (PDF + Test Bank docx).
