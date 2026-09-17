"""Complete the bibliography entries that print no journal, volume or page.

Each of them carries its DOI in the printed entry, so the record is fetched from
CrossRef and matched back by DOI rather than by position: removing one uncited
entry shifts every later number, and matching by DOI is immune to that.

Writes 01_data/audit_reference_records.json, keyed by DOI. Nothing is taken from
memory; an entry whose record does not resolve completely is reported and left
alone.
"""
import io
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = "/anvil/projects/x-mat260059/Inverse_Design_of_Bioceramics_by_Machine_Learning"
DOIS = [
    "10.1016/j.jobcr.2019.04.004", "10.1177/0022034520915714",
    "10.1155/2021/9751564", "10.7717/peerj.17793", "10.14260/jemds/2021/399",
    "10.3389/froh.2024.1442100", "10.3389/frai.2024.1427517",
    "10.3390/dj13050198", "10.3390/bioengineering11121267",
    "10.1177/0022034517720658", "10.1016/j.jdsr.2019.09.004",
    "10.1111/j.1834-7819.2010.01296.x", "10.1039/d0mh01451f",
    "10.1557/jmr.2020.43", "10.3390/ma16062166", "10.4103/jpbs.jpbs_1290_23",
    "10.1111/jerd.12566", "10.3390/jfb14080431",
    "10.1016/j.prosdent.2021.05.008", "10.1177/20552076241291345",
    "10.12659/msm.946676", "10.14260/jemds/2021/265",
]
TEX = os.path.join(ROOT, "04_manuscript", "main_revision_v1.tex")
tex = io.open(TEX, encoding="utf-8").read()

out = {}
for doi in DOIS:
    # the printed DOI escapes underscores for LaTeX, so compare unescaped
    if doi.lower() not in tex.lower().replace("\\_", "_"):
        print("  %-36s NOT PRESENT in the bibliography" % doi, flush=True)
        continue
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "bioceramics-revision/1.0 (mailto:rahma103@purdue.edu)"})
        d = json.load(urllib.request.urlopen(req, timeout=40))["message"]
    except Exception as exc:
        print("  %-36s LOOKUP FAILED: %s" % (doi, exc), flush=True)
        continue
    jour = (d.get("container-title") or [""])[0]
    vol = d.get("volume", "")
    page = d.get("page", "") or d.get("article-number", "")
    if page and "-" in page:
        page = page.split("-")[0]
    if not page and d.get("alternative-id"):
        # some journals carry only an article number, printed with an e prefix
        page = "e" + str(d["alternative-id"][0])
    if not (jour and vol and page):
        print("  %-36s incomplete: journal=%r volume=%r page=%r"
              % (doi, jour, vol, page), flush=True)
        continue
    out[doi] = dict(journal=jour, volume=vol, page=page)
    print("  %-36s %s %s, %s" % (doi, jour, vol, page), flush=True)
    time.sleep(0.4)

io.open(os.path.join(ROOT, "01_data", "audit_reference_records.json"), "w").write(
    json.dumps(out, indent=1, sort_keys=True))
print("\nresolved %d of %d" % (len(out), len(DOIS)), flush=True)
print("REFERENCE_RECORDS_DONE", flush=True)
