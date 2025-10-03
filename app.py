# app.py
import os
import time
import io
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from flask import (
    Flask,
    render_template,
    request,
    Response,
    send_file,
    jsonify,
    render_template_string,
)
from dotenv import load_dotenv

# Optional OpenAI client - only used if OPENAI_API_KEY present
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("website-health")

# Flask app - templates folder should contain your HTML UI files
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret")

# OpenAI setup (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set — /ai-insights will return an error until set.")
    else:
        logger.info("openai package not available — /ai-insights disabled.")

# Hard-coded list (replace or manage via admin UI later)
HARD_CODED_SITES = [
    "https://admin.missio.io",
    "https://admin.msplink.com",
    "https://sellers.agfolks.com",
    "https://portal.agfolks.com/portal/admin/",
    "https://www.agfolks.com/",
    "https://www.insurenex.com/",
    "https://virtualcarefl.com/",
    "https://www.wiacademy.org/main/",
    "https://washington-international-academy.mymissio.com",
    "https://maxjewelers.com/",
    "https://dreammakerclub.org",
    "https://www.americanconspiraciesandcover-ups.com",
    "https://shaaslaw.com/",
    "https://egcap.com",
    "https://prostarpayroll.com/main/",
    "https://tatyanaluxuryhomes.com/",
    "https://pbnaonline.org/",
    "https://www.sarahkrippner.com",
    "https://www.pattystaco.com",
    "https://www.thespotsportsbarandgrill.com",
    "https://www.gtfmarket.com/",
    "https://www.jamiesangels.org",
    "https://one-haus.com/",
    "https://hireintelligently.com/",
    "https://www.networkhandlers.com/",
    "https://missio.io/",
    "https://adminv2.missio.io/",
    "https://www.chnany.com/",
    "https://hbtv.missio.io/",
    "https://www.rhemafi.com/",
    "https://www.rhemawofi.org/",
    "https://www.unleashyourbold.com/",
    "https://awgenterprises.pro/",
    "https://www.bridgebuildersfcs.org/",
    "https://yescoach4u.com/",
    "https://mysisterskeephernetwork.org/",
    "https://testing.missio.io/",
    "https://shophbtv.com/",
    "https://www.themountbradley.org/",
    "https://entradapiano.com/",
    "https://www.seedsthatgrowinc.org/",
    "http://hriq.io/",
    "https://teachitpro.com",
]

# In-memory state (keeps last run for /results-table, /results-json, /download-pdf, /ai-insights)
last_results = {}  # { site: site_result }
last_run_started = None
last_run_finished = None
last_run_mode = None  # 'single' or 'bulk'

# -------------------------
# Utility helpers
# -------------------------
def _same_domain(base_url: str, link_url: str) -> bool:
    """Return True if link_url is same domain as base_url (or a relative link)."""
    try:
        b = urlparse(base_url).netloc
        l = urlparse(link_url).netloc
        return (l == "" or l == b)
    except Exception:
        return False

def _normalize_path(base: str, href: str) -> str:
    """Return a normalized absolute URL for an href relative to base."""
    try:
        return urljoin(base, href)
    except Exception:
        return href

# -------------------------
# Main analyzer
# -------------------------
def analyze_site(url: str, timeout_html=15, timeout_asset=8):
    """
    Analyze single page:
      - fetch HTML, measure fetch time
      - collect links/images/videos
      - check internal link targets (HEAD/GET) for broken links
      - return a dict containing issues and lists useful for frontend:
         { site, timing_seconds, links_total, images_total, videos_total,
           issues: {critical, major, minor}, internal_targets: [urls], child_pages: [paths] }
    """
    start = time.perf_counter()
    site_result = {
        "site": url,
        "timing_seconds": None,
        "links_total": 0,
        "images_total": 0,
        "videos_total": 0,
        "issues": {"critical": [], "major": [], "minor": []},
        # additional useful data for UI
        "internal_targets": [],  # absolute URLs this page links to (same domain)
        "child_pages": [],       # normalized internal paths (for review)
        "top_backlinks": [],     # to be computed later (populated at run end)
    }

    try:
        resp = requests.get(url, timeout=timeout_html, allow_redirects=True)
        site_result["timing_seconds"] = resp.elapsed.total_seconds() if resp is not None else None

        if resp.status_code != 200:
            site_result["issues"]["critical"].append(f"❌ Site unreachable ({resp.status_code})")
            return site_result

        html = resp.text or ""
        soup = BeautifulSoup(html, "html.parser")

        # quick raw markers
        lower = html.lower()
        if "<?php" in html or "undefined index" in lower or "stack trace" in lower:
            site_result["issues"]["critical"].append("⚠️ Raw/diagnostic code visible on page.")

        # Links (a[href])
        anchors = soup.find_all("a")
        links = [a.get("href") for a in anchors]
        site_result["links_total"] = len(links)

        for href in links:
            if not href:
                site_result["issues"]["minor"].append("⚠️ <a> without href")
                continue
            full = _normalize_path(url, href)
            # only check internal links for speed & relevance
            if not _same_domain(url, full):
                continue
            site_result["internal_targets"].append(full)
            # also collect child page relative path for UI
            parsed = urlparse(full)
            path = parsed.path or "/"
            if path not in site_result["child_pages"]:
                site_result["child_pages"].append(path)
            # check target status
            try:
                r = requests.head(full, timeout=timeout_asset, allow_redirects=True)
                # some servers block HEAD
                if r.status_code >= 400 or r.status_code in (405, 409):
                    r = requests.get(full, timeout=timeout_asset, allow_redirects=True)
                if r.status_code >= 400:
                    sev = "critical" if r.status_code in (500, 502, 503, 504) else "major"
                    site_result["issues"][sev].append(f"❌ Broken link ({r.status_code}): {full}")
            except Exception as e:
                site_result["issues"]["major"].append(f"❌ Link error: {full} → {e}")

        # Images
        imgs = [img.get("src") for img in soup.find_all("img")]
        site_result["images_total"] = len(imgs)
        for src in imgs:
            if not src:
                site_result["issues"]["minor"].append("⚠️ <img> without src")
                continue
            full = _normalize_path(url, src)
            try:
                r = requests.head(full, timeout=timeout_asset, allow_redirects=True)
                if r.status_code >= 400 or r.status_code in (405, 409):
                    # fallback
                    r = requests.get(full, timeout=timeout_asset, allow_redirects=True, stream=True)
                if r.status_code >= 400:
                    site_result["issues"]["major"].append(f"🖼️ Image not loading ({r.status_code}): {full}")
            except Exception as e:
                site_result["issues"]["major"].append(f"🖼️ Image error: {full} → {e}")

        # Videos
        videos = []
        for v in soup.find_all("video"):
            if v.get("src"):
                videos.append(v.get("src"))
            for s in v.find_all("source"):
                if s.get("src"):
                    videos.append(s.get("src"))
        site_result["videos_total"] = len(videos)
        for vs in videos:
            full = _normalize_path(url, vs)
            try:
                r = requests.head(full, timeout=timeout_asset, allow_redirects=True)
                if r.status_code >= 400 or r.status_code in (405, 409):
                    r = requests.get(full, timeout=timeout_asset, allow_redirects=True, stream=True)
                if r.status_code >= 400:
                    site_result["issues"]["major"].append(f"🎬 Video not loading ({r.status_code}): {full}")
            except Exception as e:
                site_result["issues"]["major"].append(f"🎬 Video error: {full} → {e}")

    except Exception as e:
        site_result["issues"]["critical"].append(f"❌ Site fetch failed: {e}")

    elapsed = time.perf_counter() - start
    # timing info is already in site_result['timing_seconds']
    return site_result

def _format_site_log(sr, started_at):
    """Readable lines for SSE stream log"""
    elapsed = time.perf_counter() - started_at
    lines = []
    lines.append(f"⏱️ Elapsed: {elapsed:0.1f}s")
    if sr.get("timing_seconds") is not None:
        lines.append(f"⏱️ HTML fetch: {sr['timing_seconds']:.2f}s")
    lines.append(f"🔗 Links: {sr['links_total']} · 🖼️ Images: {sr['images_total']} · 🎬 Videos: {sr['videos_total']}")
    for cat in ("critical", "major", "minor"):
        for msg in sr["issues"].get(cat, []):
            lines.append(msg)
    if not any(sr["issues"].values()):
        lines.append("✅ No issues found.")
    return lines

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    """Serve main UI (put your index UI in templates/index.html)"""
    return render_template("index.html")

@app.route("/scan")
def scan():
    """SSE single-site scan streaming"""
    global last_results, last_run_started, last_run_finished, last_run_mode
    url = request.args.get("url", "").strip()
    if not url:
        return Response("Missing ?url", status=400)

    def stream():
        global last_results, last_run_started, last_run_finished, last_run_mode
        last_run_started = time.time()
        started_at = time.perf_counter()
        yield f"data: 🔍 Scanning {url}...\n\n"
        sr = analyze_site(url)
        # store result
        last_results = {url: sr}
        # compute top_backlinks later after run(s)
        for line in _format_site_log(sr, started_at):
            yield f"data: {line}\n\n"
        total_taken = time.perf_counter() - started_at
        yield f"data: ✅ Finished {url} in {total_taken:0.1f}s\n\n"
        last_run_mode = "single"
        last_run_finished = time.time()
        # After finishing streaming, compute top_backlinks based on current last_results
        _compute_top_backlinks()
        yield "data: ---DONE---\n\n"

    return Response(stream(), mimetype="text/event-stream")

@app.route("/bulk-scan")
def bulk_scan():
    """SSE bulk scan for HARD_CODED_SITES (sequential)"""
    global last_results, last_run_started, last_run_finished, last_run_mode
    sites = HARD_CODED_SITES[:]
    total_start = time.perf_counter()
    last_run_started = time.time()

    def stream():
        nonlocal sites, total_start
        results = {}
        n = len(sites)
        for i, site in enumerate(sites, start=1):
            site_start = time.perf_counter()
            yield f"data: 🌐 Checking {site} ({i}/{n})...\n\n"
            sr = analyze_site(site)
            # store temporarily
            results[site] = sr
            for line in _format_site_log(sr, site_start):
                yield f"data: {line}\n\n"
            took = time.perf_counter() - site_start
            yield f"data: ✅ Finished {site} in {took:0.1f}s\n\n"
        total_took = time.perf_counter() - total_start
        yield f"data: 🧮 Total time: {total_took:0.1f}s\n\n"
        # Save last run
        last_results = results
        last_run_mode = "bulk"
        last_run_finished = time.time()
        # compute backlinks (internal) across results
        _compute_top_backlinks()
        yield "data: ---DONE---\n\n"

    return Response(stream(), mimetype="text/event-stream")

@app.route("/results-json")
def results_json():
    """Return last_results as JSON for dynamic UI rendering"""
    if not last_results:
        return jsonify({"error": "No results yet. Run a scan first."}), 400
    return jsonify(last_results)

@app.route("/results-table")
def results_table():
    """Render a nicer table page (templates/results.html expected)"""
    if not last_results:
        return "<h3>No results yet. Run a scan first.</h3>"

    # If you have results.html template, render it; otherwise fallback to simple view
    try:
        return render_template("results.html", results=last_results)
    except Exception:
        return "<pre>" + json.dumps(last_results, indent=2) + "</pre>"

@app.route("/download-pdf")
def download_pdf():
    """Return results as a simple downloadable text file (could be replaced by a proper PDF generator)"""
    if not last_results:
        return "<h3>No results yet. Run a scan first.</h3>"

    buf = io.StringIO()
    buf.write("Website Health Report\n")
    buf.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if last_run_mode:
        buf.write(f"Mode: {last_run_mode}\n")
    buf.write("\n")
    k=ai_insights()
    for site, sr in last_results.items():
        buf.write(f"== {site} ==\n")
        if sr.get("timing_seconds") is not None:
            buf.write(f"HTML fetch: {sr['timing_seconds']:.2f}s\n")
        buf.write(f"Links: {sr['links_total']} | Images: {sr['images_total']} | Videos: {sr['videos_total']}\n")
        for cat in ("critical", "major", "minor"):
            issues = sr["issues"].get(cat, [])
            if issues:
                buf.write(f"\n{cat.upper()} ({len(issues)}):\n")
                for m in issues:
                    buf.write(f"  - {m}\n")
        if k:
            buf.write(" AI summary:\n")
            for tb in k :
                buf.write(f"  - {tb['url']} ({tb['count']})\n")
        buf.write("\n")        
        # child pages
        if sr.get("child_pages"):
            buf.write("\nChild pages:\n")
            for cp in sr["child_pages"]:
                buf.write(f"  - {cp}\n")
        # backlinks (top_backlinks)
        if sr.get("top_backlinks"):
            buf.write("\nTop Backlinks (internal):\n")
            for tb in sr["top_backlinks"][:20]:
                buf.write(f"  - {tb['url']} ({tb['count']})\n")
        buf.write("\n")
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, as_attachment=True, download_name="website_health_report.txt", mimetype="text/plain")

# -------------------------
# Backlink computation
# -------------------------
def _compute_top_backlinks():
    """
    Compute internal inbound link counts across last_results and attach:
      - for each site result: sr['top_backlinks'] = [ {'url': path, 'count': n}, ... ]
    We compute counts by mapping internal_targets across all pages, then counting.
    """
    global last_results
    # collect all internal targets per site
    target_counts = Counter()
    per_site_targets = defaultdict(list)
    for site, sr in last_results.items():
        targets = sr.get("internal_targets", []) or []
        per_site_targets[site] = targets
        for t in targets:
            # only count internal path (host + path)
            parsed = urlparse(t)
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}"
            target_counts[normalized] += 1

    # build top_backlinks per site (only include targets belonging to same site host)
    for site, sr in last_results.items():
        parsed_site = urlparse(site)
        host = parsed_site.netloc
        # filter counts to those that belong to this host
        host_counts = []
        for target_url, cnt in target_counts.items():
            p = urlparse(target_url)
            if p.netloc == host:
                host_counts.append({"url": target_url, "count": cnt})
        # sort descending
        host_counts.sort(key=lambda x: x["count"], reverse=True)
        sr["top_backlinks"] = host_counts
        # ensure child_pages exists
        if "child_pages" not in sr:
            sr["child_pages"] = []

# -------------------------
# AI Insights
# -------------------------
@app.route("/ai-insights")
def ai_insights():
    """Return AI generated summary + prioritized recommendations based on last_results."""
    if not last_results:
        return jsonify({"error": "No last results available. Run a scan first."}), 400
    if not openai_client:
        return jsonify({"error": "OpenAI API key not configured on server."}), 500

    # Build compact summary lines
    lines = []
    for site, sr in last_results.items():
        crit = len(sr["issues"].get("critical", []))
        maj = len(sr["issues"].get("major", []))
        minr = len(sr["issues"].get("minor", []))
        speed = f"{sr['timing_seconds']:.2f}s" if sr.get("timing_seconds") else "—"
        lines.append(f"{site} | speed:{speed} | critical:{crit} major:{maj} minor:{minr}")

    prompt = f"""
You are an expert website QA engineer. Given the following scan summary lines (one per site), provide:
1) A 2-3 sentence executive summary about the overall health.
2) A prioritized list of up to 6 concrete recommendations (each 1 line) with priority tags High/Medium/Low.

Scan summary:
{chr(10).join(lines)}

Return only valid JSON with two fields: "summary" (string) and "recommendations" (array of strings).
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful web QA expert."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()

        try:
            parsed = json.loads(text)
            summary = parsed.get("summary", "")
            recommendations = parsed.get("recommendations", [])
        except Exception:
            # fallback: treat whole text as summary
            summary = text
            recommendations = []
        return jsonify({"summary_text": text})


    except Exception as e:
        logger.exception("OpenAI call failed")
        return jsonify({"error": f"OpenAI request failed: {e}"}), 500

@app.route("/ai-chat", methods=["POST"])
def ai_chat():
    """Chat with the AI based on scan results."""
    if not last_results:
        return jsonify({"error": "Run a scan first."}), 400
    if not openai_client:
        return jsonify({"error": "OpenAI API key missing."}), 500

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message required"}), 400

    # Build context from last scan
    lines = []
    for site, sr in last_results.items():
        crit = len(sr["issues"].get("critical", []))
        maj = len(sr["issues"].get("major", []))
        minr = len(sr["issues"].get("minor", []))
        speed = f"{sr['timing_seconds']:.2f}s" if sr.get("timing_seconds") else "—      "
        lines.append(f"{site} | speed:{speed} | critical:{crit} major:{maj} minor:{minr}")

    context = "\n".join(lines)

    prompt = f"""
You are a helpful website QA consultant. You have this scan context:

{context}

Now the user asks: "{user_message}"

👉 Respond in **clear, concise bullet points**.  
👉 Keep it short, no more than 5 bullets.  
👉 Make it visually engaging (✅, ⚡, ❗ etc. where relevant).  
👉 Avoid raw JSON or code unless specifically asked.  

Give a practical, concise, and actionable response.
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful QA consultant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.4,
        )
        answer = resp.choices[0].message.content.strip()
        return jsonify({"response": answer})
    except Exception as e:
        logger.exception("AI chat failed")
        return jsonify({"error": f"AI chat failed: {e}"}), 500

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
