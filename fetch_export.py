#!/usr/bin/env python3
# ================== PACKAGES ==================
import asyncio, json, os, re, tempfile
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ================== ENV & SETTINGS ==================
def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

ARMS_USERNAME = _req("ARMS_USERNAME")
ARMS_PASS     = _req("ARMS_PASS")
ARMS_URL      = _req("ARMS_URL")
HEADLESS      = os.getenv("HEADLESS", "true").lower() in ("1", "true", "yes")

# ================== UTILS / CACHE ==================
CACHE_PATH = Path(__file__).with_name(".exports_cache.json")

def _read_cache():
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}

def _write_cache(d):
    try:
        CACHE_PATH.write_text(json.dumps(d, indent=2))
    except Exception:
        pass

def _rx_exact(s: str):
    return re.compile(rf"^\s*{re.escape(s)}\s*$", re.I)

def _rx_startswith(s: str):
    return re.compile(rf"^\s*{re.escape(s)}\b", re.I)

def _layout_tokens(layout_text: str):
    toks = re.findall(r"[A-Za-z0-9]+", layout_text.lower())
    stop = {"the","of","for","and","a","an","to","by","in","on"}
    return [t for t in toks if t not in stop]

def _filename_matches_layout(fn: str, tokens):
    s = re.sub(r"[^a-z0-9]+", " ", fn.lower())
    return all(t in s for t in tokens)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ================== CONFIG HELPERS ==================
def _get_grad_year(exp: Dict) -> Optional[str]:
    sel = (exp.get("filters", {})
              .get("gradYear", {})
              .get("selector", []))
    if isinstance(sel, str):
        m = re.search(r"\b(19|20)\d{2}\b", sel)
        return m.group(0) if m else sel.strip()
    if isinstance(sel, list) and sel:
        return str(sel[0])
    return None

def _get_acs_vals(exp: Dict) -> List[str]:
    sel = (exp.get("filters", {})
              .get("ACS Rank", {})
              .get("selector", []))
    if isinstance(sel, str):
        return [s.strip() for s in re.split(r"[,/|]+", sel) if s.strip()]
    if isinstance(sel, list):
        return sel
    return []

# ================== NAV / FILTER UI ==================
async def click_recruiting_recruits(page):
    """From Dashboard left nav, open Recruiting → Recruits."""
    try:
        chevron = page.locator("button, [role='button']").filter(
            has_text=re.compile(r"^\s*Close Menu|Open Menu\s*$", re.I)
        ).first
        if await chevron.count():
            try:
                await chevron.click(timeout=800)
            except:
                pass
    except:
        pass

    for loc in [
        page.get_by_role("link", name=_rx_exact("Recruiting")).first,
        page.get_by_role("button", name=_rx_exact("Recruiting")).first,
        page.locator("nav,aside").get_by_text(_rx_exact("Recruiting")).first,
        page.locator("nav svg use[href*='recruiting-icon'], nav svg use[xlink\\:href*='recruiting-icon']").first
    ]:
        try:
            await loc.scroll_into_view_if_needed()
            await loc.click(timeout=3000)
            break
        except:
            continue
    else:
        raise RuntimeError("Could not find 'Recruiting' in left navigation.")

    for loc in [
        page.get_by_role("link",  name=_rx_exact("Recruits")).first,
        page.get_by_role("menuitem", name=_rx_exact("Recruits")).first,
        page.get_by_text(_rx_exact("Recruits")).first,
    ]:
        try:
            await loc.click(timeout=4000)
            break
        except:
            continue
    else:
        raise RuntimeError("Could not click ‘Recruits’ in the flyout.")

    await page.wait_for_load_state("networkidle")

async def _expand_section(scope, title_regex):
    try:
        hdr = scope.get_by_role("button", name=title_regex).first
        await hdr.wait_for(timeout=1200)
        expanded = await hdr.get_attribute("aria-expanded")
        if expanded is not None and expanded.lower() == "false":
            await hdr.click()
            await scope.wait_for_load_state("networkidle")
            await asyncio.sleep(0.1)
            return
    except:
        pass
    try:
        hdr2 = scope.locator(".mat-expansion-panel-header").filter(has=scope.get_by_text(title_regex)).first
        await hdr2.wait_for(timeout=1200)
        classes = (await hdr2.get_attribute("class")) or ""
        if "mat-expanded" not in classes:
            await hdr2.click()
            await asyncio.sleep(0.1)
    except:
        pass

async def _click_link_in_section(scope, section_title_rx, link_text_rx):
    sec = scope.locator("section,div,aside").filter(has=scope.get_by_text(section_title_rx)).first
    try:
        await sec.wait_for(timeout=1000)
    except:
        sec = scope
    for loc in [sec.get_by_role("link", name=link_text_rx).first, sec.get_by_text(link_text_rx).first]:
        try:
            await loc.wait_for(timeout=600)
            await loc.click()
            await asyncio.sleep(0.05)
            return True
        except:
            continue
    return False

async def _scroll_until_visible(scope, regex, max_steps=20):
    container = scope.locator(".mat-drawer-content, .mat-sidenav-content, .cdk-virtual-scroll-viewport").first
    for _ in range(max_steps):
        try:
            el = scope.get_by_text(regex).first
            await el.scroll_into_view_if_needed()
            await el.wait_for(timeout=300)
            return True
        except:
            try:
                await container.evaluate("(el)=>el.scrollBy(0,300)")
            except:
                await scope.evaluate("()=>window.scrollBy(0,300)")
            await asyncio.sleep(0.05)
    return False

async def ensure_checkbox_checked(scope, name_regex):
    host = scope.locator("mat-checkbox").filter(has=scope.get_by_text(name_regex)).first
    try:
        if await host.count():
            classes = (await host.get_attribute("class")) or ""
            if "mat-checkbox-checked" in classes:
                return
            for tgt_sel in [".mat-checkbox-inner-container", "label", ".mat-checkbox-layout"]:
                tgt = host.locator(tgt_sel)
                try:
                    await tgt.scroll_into_view_if_needed()
                    await tgt.click()
                    return
                except:
                    continue
            await host.click(force=True)
            return
    except:
        pass
    try:
        lbl = scope.get_by_label(name_regex).first
        await lbl.scroll_into_view_if_needed()
        try:
            await lbl.check()
        except:
            await lbl.click()
        return
    except:
        pass
    await scope.get_by_text(name_regex).first.click(force=True)

async def find_filters_scope(page):
    try:
        await page.get_by_text(_rx_exact("Grad. Year")).first.wait_for(timeout=1200)
        return page
    except:
        pass
    for fr in page.frames:
        try:
            await fr.get_by_text(_rx_exact("Grad. Year")).first.wait_for(timeout=800)
            return fr
        except:
            continue
    return page

async def apply_filters(scope, grad_year: Optional[str],
                        statuses: Optional[List[str]] = None,
                        status_section_label: str = "Status"):
    await _expand_section(scope, _rx_exact(status_section_label))
    await _expand_section(scope, _rx_exact("Grad. Year"))

    # Status/ACS Rank
    if statuses:
        await _click_link_in_section(scope, _rx_exact(status_section_label), _rx_exact("none"))
        for s in statuses:
            await ensure_checkbox_checked(scope, _rx_exact(s))
    else:
        await _click_link_in_section(scope, _rx_exact(status_section_label), _rx_exact("all"))

    # Grad Year
    if grad_year:
        await _click_link_in_section(scope, _rx_exact("Grad. Year"), _rx_exact("none"))
        rx_year = _rx_startswith(grad_year)
        await _scroll_until_visible(scope, rx_year)
        await ensure_checkbox_checked(scope, rx_year)

# ================== EXPORT UI ==================
async def open_right_kebab_and_click_export(page):
    trigger_selectors = [
        "button[aria-haspopup='menu']",
        "button[aria-label*='menu' i]",
        "button[title*='menu' i]",
        "div[role='toolbar'] button",
        "header button",
    ]
    triggers = page.locator(",".join(trigger_selectors))

    bulk_icon = page.locator("mat-icon[aria-label*='Bulk Update Menu' i]").first
    try:
        if await bulk_icon.count():
            btn_from_icon = bulk_icon.locator("xpath=ancestor::button[1]")
            try:
                # playwright <1.49 may not have union; fall back if needed
                triggers = triggers.union(btn_from_icon)  # type: ignore
            except Exception:
                pass
    except:
        pass

    if await triggers.count() == 0:
        raise RuntimeError("Hamburger/3-line menu not found")

    right_idx, right_x = 0, -1
    for i in range(await triggers.count()):
        el = triggers.nth(i)
        try:
            await el.wait_for(timeout=1500)
            box = await el.bounding_box()
            if box and box["y"] < 4000 and box["x"] > right_x:
                right_x, right_idx = box["x"], i
        except:
            continue
    menu_btn = triggers.nth(right_idx)

    async def _open_menu_and_click_export():
        await menu_btn.scroll_into_view_if_needed()
        await menu_btn.click(force=True)
        panel = page.locator(".cdk-overlay-pane:has(.mat-menu-content), [role='menu']").last
        await panel.wait_for(timeout=3000)

        candidates = [
            panel.locator('[data-cy="export"]').first,
            panel.locator('button[role="menuitem"][data-cy="export"]').first,
            panel.get_by_role("menuitem", name=_rx_exact("Export")).first,
            panel.get_by_role("button",   name=_rx_exact("Export")).first,
            panel.locator(".mat-menu-content .mat-menu-item:has-text('Export')").first,
            panel.locator("text=Export").first,
        ]
        for el in candidates:
            try:
                if await el.count() and await el.is_visible():
                    await el.scroll_into_view_if_needed()
                    await el.click(timeout=1500)
                    await page.wait_for_load_state("networkidle")
                    return True
            except:
                continue

        try:
            items = panel.locator("[role='menuitem'], .mat-menu-content .mat-menu-item, .mat-menu-content a")
            n = await items.count()
            if n:
                texts = []
                for i in range(min(n, 20)):
                    try:
                        t = (await items.nth(i).inner_text()).strip()
                        if t:
                            texts.append(t)
                    except:
                        pass
                if texts:
                    print("[debug] hamburger menu items:", " | ".join(texts))
        except:
            pass

        try:
            await page.keyboard.press("Escape")
        except:
            pass
        await asyncio.sleep(0.15)
        return False

    for _ in range(3):
        if await _open_menu_and_click_export():
            return
    raise RuntimeError("Export option not found after opening menu")

async def open_export_and_start_job(layout_text: str, page):
    dropdown = None
    for loc in [
        page.locator("#exportLayout"),
        page.get_by_role("combobox").filter(has_text=re.compile("Export Layout|Layout", re.I)).first,
        page.get_by_role("button", name=re.compile(r"Export Layout|Select layout|Layout", re.I)).first,
        page.get_by_label(re.compile(r"Export Layout|Layout", re.I)).first,
    ]:
        try:
            await loc.wait_for(timeout=5000)
            dropdown = loc
            break
        except:
            continue
    if not dropdown:
        raise RuntimeError("Export modal: layout dropdown not found.")

    await dropdown.scroll_into_view_if_needed()
    await dropdown.click()

    picked = False
    for finder in [
        lambda: page.get_by_role("option",   name=_rx_exact(layout_text)).first,
        lambda: page.get_by_role("menuitem", name=_rx_exact(layout_text)).first,
        lambda: page.get_by_text(            _rx_exact(layout_text)).first,
    ]:
        try:
            await finder().click(timeout=5000)
            picked = True
            break
        except:
            continue
    if not picked:
        raise RuntimeError(f"Export modal: layout '{layout_text}' not found.")

    export_btn_candidates = [
        page.get_by_role("button", name=re.compile(r"^\s*Export\b.*", re.I)).first,
        page.locator("button[type='submit']").first,
        page.locator("button.k-button--primary, button.mat-primary").filter(
            has_text=re.compile(r"^\s*Export\b", re.I)
        ).first,
        page.get_by_text(re.compile(r"^\s*Export\b.*", re.I)).first,
    ]
    for btn in export_btn_candidates:
        try:
            await btn.scroll_into_view_if_needed()
            await page.wait_for_timeout(200)
            await btn.click(timeout=5000)
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(0.5)
            return
        except:
            continue
    raise RuntimeError("Export modal: could not find/click the Export button.")

async def maybe_go_to_exports_prompt(page):
    for finder in [
        lambda: page.get_by_role("button", name=re.compile(r"^\s*Take me to Exports page\s*$", re.I)).first,
        lambda: page.get_by_text(re.compile(r"^\s*Take me to Exports page\s*$", re.I)).first,
        lambda: page.get_by_role("button", name=re.compile(r"^\s*Go to Exports\s*$", re.I)).first,
        lambda: page.get_by_role("link",   name=re.compile(r"^\s*Go to Exports\s*$", re.I)).first,
        lambda: page.get_by_text(re.compile(r"^\s*Go to Exports\s*$", re.I)).first,
        lambda: page.get_by_role("button", name=re.compile(r"^\s*Go to Export(s)? Page\s*$", re.I)).first,
        lambda: page.get_by_text(re.compile(r"^\s*Go to Export(s)? Page\s*$", re.I)).first,
    ]:
        try:
            await finder().click(timeout=2000)
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(0.3)
            return True
        except:
            continue
    return False

async def disable_auto_refresh_if_present(page):
    try:
        block = page.locator("text=This page will auto-refresh").first
        await block.wait_for(timeout=2000)
        container = block.locator("xpath=..")
        toggle = container.locator("button, [role='button']").filter(has=page.locator("svg")).first
        pressed = await toggle.get_attribute("aria-pressed")
        if pressed is None or pressed.lower() == "true":
            await toggle.click(timeout=1500)
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(0.25)
    except:
        pass

async def fetch_latest_export_from_admin(page, layout_text: str, timeout_s: int = 180, skip_if_same=True) -> pd.DataFrame:
    tokens = _layout_tokens(layout_text)

    url = page.url.lower()
    if "admin" not in url or "export" not in url:
        for step in [
            lambda: page.get_by_text(_rx_exact("Administration")).first.click(timeout=3000),
            lambda: page.get_by_role("link", name=re.compile(r"Administration", re.I)).first.click(timeout=3000),
        ]:
            try:
                await step()
                await page.wait_for_load_state("networkidle")
                break
            except:
                pass
        for step in [
            lambda: page.get_by_text(_rx_exact("Exports")).first.click(timeout=3000),
            lambda: page.get_by_role("link", name=re.compile(r"Exports", re.I)).first.click(timeout=3000),
        ]:
            try:
                await step()
                await page.wait_for_load_state("networkidle")
                break
            except:
                pass

    await disable_auto_refresh_if_present(page)

    try:
        submit_hdr = page.locator("table thead th").filter(
            has=page.get_by_text(re.compile(r"^\s*Submit\s*Date\s*$", re.I))
        ).first
        await submit_hdr.click(timeout=1500)
        await page.wait_for_load_state("networkidle")
        await submit_hdr.click(timeout=1500)
        await page.wait_for_load_state("networkidle")
    except:
        pass

    file_col_idx = None
    try:
        ths = page.locator("table thead th")
        n_th = await ths.count()
        for i in range(n_th):
            t = (await ths.nth(i).inner_text()).strip().lower()
            if "file" in t and "data" in t:
                file_col_idx = i
                break
    except:
        pass

    async def _find_newest_complete():
        body_rows = page.locator("table tbody tr")
        n = await body_rows.count()
        for i in range(n):
            row = body_rows.nth(i)
            try:
                await row.get_by_text(re.compile(r"\bComplete(d)?\b", re.I)).first.wait_for(timeout=250)
            except:
                continue

            if file_col_idx is not None:
                cell = row.locator("td").nth(file_col_idx)
                link = cell.locator("a").first
            else:
                link = row.locator("a").first

            try:
                fn = (await link.inner_text()).strip()
            except:
                continue

            if not fn or not _filename_matches_layout(fn, tokens):
                continue

            return row, link, fn
        return None

    end = asyncio.get_event_loop().time() + timeout_s
    found = None
    while asyncio.get_event_loop().time() < end:
        found = await _find_newest_complete()
        if found:
            break
        await asyncio.sleep(1.0)

    if not found:
        raise RuntimeError(f"Exports: no COMPLETE file found for layout '{layout_text}' within timeout.")

    row, link_el, filename = found

    cache = _read_cache() if skip_if_same else {}
    if skip_if_same and cache.get(layout_text) == filename:
        print(f"[info] latest file for '{layout_text}' already processed: {filename}")
        return pd.DataFrame()
    if skip_if_same:
        cache[layout_text] = filename
        _write_cache(cache)

    async with page.expect_download() as dl_ctx:
        await link_el.click()
    download = await dl_ctx.value

    with tempfile.TemporaryDirectory() as td:
        path = await download.path()
        if path is None:
            save_to = os.path.join(td, download.suggested_filename or filename or "export.csv")
            await download.save_as(save_to)
            path = save_to
        try:
            return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, dtype=str)

async def start_export_from_admin(layout_text: str, page):
    for step in [
        lambda: page.get_by_role("link", name=re.compile(r"Administration", re.I)).first.click(timeout=3000),
        lambda: page.get_by_text(re.compile(r"^\s*Administration\s*$", re.I)).first.click(timeout=3000),
    ]:
        try:
            await step()
            await page.wait_for_load_state("networkidle")
            break
        except:
            pass
    for step in [
        lambda: page.get_by_role("link", name=re.compile(r"Exports", re.I)).first.click(timeout=3000),
        lambda: page.get_by_text(re.compile(r"^\s*Exports\s*$", re.I)).first.click(timeout=3000),
    ]:
        try:
            await step()
            await page.wait_for_load_state("networkidle")
            break
        except:
            pass

    for sel in [
        "button[aria-label*='Menu']",
        "button[title*='Menu']",
        "button:has(svg)",
        "button:has-text('≡')",
        "button:has(.kebab), button:has(.hamburger)",
    ]:
        try:
            btn = page.locator(sel).first
            await btn.wait_for(timeout=5000)
            await btn.scroll_into_view_if_needed()
            await btn.click()
            await page.wait_for_timeout(1000)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Hamburger/3-line menu not found")

    for sel in [
        "text=Export",
        "button:has-text('Export')",
        "div[role='menu'] >> text=Export",
    ]:
        try:
            export_btn = page.locator(sel).first
            await export_btn.wait_for(timeout=5000)
            await export_btn.click()
            await page.wait_for_load_state("networkidle")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Export option not found after opening menu")

    dropdown = None
    for loc in [
        page.locator("#exportLayout"),
        page.get_by_role("combobox").filter(has_text=re.compile("Export Layout|Layout", re.I)).first,
        page.get_by_role("button", name=re.compile(r"Export Layout|Select layout|Layout", re.I)).first,
        page.get_by_label(re.compile(r"Export Layout|Layout", re.I)).first,
    ]:
        try:
            await loc.wait_for(timeout=4000)
            await loc.click()
            dropdown = loc
            break
        except:
            continue
    if not dropdown:
        raise RuntimeError("Admin Export: layout selector not found.")

    picked = False
    for finder in [
        lambda: page.get_by_role("option",   name=_rx_exact(layout_text)).first,
        lambda: page.get_by_role("menuitem", name=_rx_exact(layout_text)).first,
        lambda: page.get_by_text(            _rx_exact(layout_text)).first,
    ]:
        try:
            el = finder()
            await el.scroll_into_view_if_needed()
            await el.click(timeout=4000)
            picked = True
            break
        except:
            continue
    if not picked:
        raise RuntimeError(f"Admin Export: layout '{layout_text}' not found.")

    for btn in [
        page.get_by_role("button", name=re.compile(r"^\s*Export\b", re.I)).first,
        page.locator("button[type='submit']").first,
        page.locator("button.k-button--primary, button.mat-primary").filter(
            has_text=re.compile(r"^\s*Export\b", re.I)
        ).first,
    ]:
        try:
            await btn.scroll_into_view_if_needed()
            await page.wait_for_timeout(200)
            await btn.click(timeout=4000)
            await page.wait_for_load_state("networkidle")
            return
        except:
            continue
    raise RuntimeError("Admin Export: could not click the final Export button.")

# ================== CORE FLOW ==================
async def do_one_export(page, exp: Dict) -> Optional[str]:
    """
    Runs one export as defined by 'exp' object from config.
    Returns run_id if a CSV was saved; None if skipped.
    """
    layout_text = (
        exp.get("export", {}).get("export_profile")
        or exp.get("export", {}).get("layoutOptionText")
        or exp.get("name", "Export").replace("_", " ")
    )

    # Navigate to Recruits & apply filters
    await click_recruiting_recruits(page)
    scope = await find_filters_scope(page)
    try:
        gy  = _get_grad_year(exp)
        acs = _get_acs_vals(exp)
        await apply_filters(scope, grad_year=gy, statuses=acs, status_section_label="ACS Rank")
    except Exception as e:
        print(f"[warn] filter step issue: {e}")

    # Start export via main menu; fallback to Admin → Exports
    try:
        await open_right_kebab_and_click_export(page)
        await open_export_and_start_job(layout_text, page)
        await maybe_go_to_exports_prompt(page)
    except Exception as e:
        print(f"[warn] hamburger path failed: {e} — falling back to Admin → Exports")
        await start_export_from_admin(layout_text, page)

    # Download latest completed export to DataFrame
    df = await fetch_latest_export_from_admin(page, layout_text, skip_if_same=False)
    if df is None or df.empty:
        print(f"[info] No new rows for '{layout_text}' (skipped).")
        return None

    # Save to repo workspace for the processing script
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("output/raw")
    _ensure_dir(out_dir)
    raw_csv = out_dir / f"export_{run_id}.csv"
    df.to_csv(raw_csv, index=False)

    manifest = {
        "run_id": run_id,
        "layout": layout_text,
        "rows": int(len(df)),
        "saved_csv": str(raw_csv),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    (out_dir / f"manifest_{run_id}.json").write_text(json.dumps(manifest, indent=2))

    # convenience for the next step
    Path("output/last_run_id.txt").write_text(run_id)
    print(f"RUN_ID={run_id}")
    print(f"[info] saved raw CSV to {raw_csv}")
    return run_id

async def login(page, url, username, password):
    await page.goto(url, wait_until="load")
    try:
        await page.get_by_label(re.compile(r"Email|Username", re.I)).first.fill(username)
    except:
        await page.locator('input[type="email"], input[name*="user" i], input[type="text"]').first.fill(username)

    try:
        btn_next = page.get_by_role("button", name=_rx_exact("Next")).first
        if await btn_next.count():
            await btn_next.click()
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(800)
    except:
        pass

    async def _find_password_locator():
        candidates = [
            page.get_by_label(re.compile(r"Password", re.I)).first,
            page.locator('input[type="password"]').first,
            page.locator('input[name*="pass" i]').first,
        ]
        for loc in candidates:
            try:
                await loc.wait_for(timeout=6000)
                return loc
            except:
                pass
        for fr in page.frames:
            candidates = [
                fr.get_by_label(re.compile(r"Password", re.I)).first,
                fr.locator('input[type="password"]').first,
                fr.locator('input[name*="pass" i]').first,
            ]
            for loc in candidates:
                try:
                    await loc.wait_for(timeout=4000)
                    return loc
                except:
                    pass
        return None

    pwd = await _find_password_locator()
    if not pwd:
        try:
            await page.keyboard.press("Tab")
            await page.wait_for_timeout(400)
            pwd = await _find_password_locator()
        except:
            pass
    if not pwd:
        raise RuntimeError("Could not find password field after waiting")

    await pwd.fill(password)

    submitted = False
    for b in [
        page.get_by_role("button", name=re.compile(r"Sign in|Log in|Login", re.I)).first,
        page.locator('button[type="submit"]').first,
    ]:
        try:
            await b.click(timeout=4000)
            submitted = True
            break
        except:
            continue
    if not submitted:
        try:
            await pwd.press("Enter")
        except:
            pass

    try:
        await asyncio.wait_for(asyncio.wait([
            page.wait_for_selector("a:has-text('Recruiting')", state="visible"),
            page.wait_for_selector("text=Administration", state="visible"),
            page.wait_for_selector("text=Exports", state="visible"),
        ], return_when=asyncio.FIRST_COMPLETED), timeout=90)
    except Exception:
        # Fallback so the flow can continue even if the above didn't resolve
        await page.wait_for_load_state("domcontentloaded")


async def run():
    cfg_path = Path(__file__).with_name("config.json")
    config = json.loads(cfg_path.read_text())

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(accept_downloads=True, viewport={"width": 1366, "height": 900})
        page = await context.new_page()

        print("[info] Logging into ARMS ...")
        await login(page, ARMS_URL, ARMS_USERNAME, ARMS_PASS)
        print("[info] Login complete.")

        # Execute exports in config
        for exp in config.get("exports", []):
            try:
                await do_one_export(page, exp)
            except Exception as e:
                print(f"[error] export failed for {exp.get('name','Unnamed')}: {e}")

        await context.close()
        await browser.close()
        print("\n[done] All exports processed.")

if __name__ == "__main__":
    asyncio.run(run())
