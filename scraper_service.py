import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import time
import re
from typing import Generator

BASE_URL = "https://www.institutdesactuaires.com"
SEARCH_URL = f"{BASE_URL}/se-documenter/memoires/memoires-d-actuariat-4651"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_session = requests.Session()


def _fetch_year(annee: int, mots_cles: list[str], mots_exclusion: list[str], timeout: int) -> list[dict]:
    unique: dict[str, dict] = {}
    for mot in mots_cles:
        try:
            r = _session.post(SEARCH_URL, data={"annee": str(annee), "keyword": mot, "submit": "Rechercher"}, headers=HEADERS, timeout=timeout)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "html.parser")
            table = soup.find("table", class_="table table-striped")
            if not table:
                continue
            for row in table.find_all("tr")[1:]:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue
                auteur = cols[0].get_text(strip=True)
                societe = cols[1].get_text(strip=True)
                annee_col = cols[2].get_text(strip=True)
                titre = cols[3].get_text(strip=True)
                if any(ex in titre.lower() for ex in mots_exclusion):
                    continue
                a_tag = next((c.find("a") for c in cols if c.find("a")), None)
                if a_tag and a_tag.get("href"):
                    href = a_tag["href"]
                    lien = SEARCH_URL + href if href.startswith("?") else (href if href.startswith("http") else BASE_URL + href)
                else:
                    lien = None
                if lien and lien not in unique:
                    unique[lien] = {"auteur": auteur, "societe": societe, "annee": annee_col, "titre": titre, "lien": lien, "resume_fr": "", "confidentiel": False, "lien_pdf": None}
            time.sleep(0.5)
        except Exception:
            continue
    return list(unique.values())


def _extract_details(url: str, timeout: int) -> dict:
    empty = {"resume_fr": "", "confidentiel": False, "lien_pdf": None}
    if not url:
        return empty
    try:
        r = _session.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return empty
        soup = BeautifulSoup(r.content, "html.parser")
        lien_pdf = None
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if ".pdf" in href.lower():
                if "/docs/mem/" in href:
                    lien_pdf = BASE_URL + href if not href.startswith("http") else href
                    break
                elif not any(x in href.lower() for x in ["grille", "charte", "guide"]):
                    lien_pdf = BASE_URL + href if not href.startswith("http") else href
        resume_fr = ""
        for b in soup.find_all("b"):
            if re.search(r"R[eé]sum[eé]", b.get_text(), re.IGNORECASE):
                parts = []
                cur = b.next_sibling
                while cur:
                    if isinstance(cur, Tag) and cur.name == "b":
                        break
                    txt = str(cur).strip() if isinstance(cur, NavigableString) else cur.get_text(" ", strip=True) if isinstance(cur, Tag) else ""
                    if txt and txt not in ["<br>", "<br/>"]:
                        parts.append(txt)
                    cur = cur.next_sibling
                resume_fr = re.sub(r"\s+", " ", " ".join(parts)).strip()
                break
        confidentiel = "confidentiel" in soup.get_text().lower()
        if lien_pdf:
            confidentiel = False
        return {"resume_fr": resume_fr[:5000], "confidentiel": confidentiel, "lien_pdf": lien_pdf}
    except Exception:
        return empty


def scrape(params: dict) -> Generator[dict, None, None]:
    """Synchronous generator yielding progress/data events for the SSE route."""
    annee_min = params["annee_min"]
    annee_max = params["annee_max"]
    mots_cles = params["mots_cles"]
    mots_exclusion = [m.lower() for m in params.get("mots_exclusion", [])]
    extract_details = params.get("extract_details", True)
    delay = params.get("delay", 2)
    timeout = params.get("timeout", 30)

    annees = list(range(annee_min, annee_max + 1))
    total_years = len(annees)
    all_memoires: list[dict] = []

    for year_idx, annee in enumerate(annees):
        yield {"type": "status", "message": f"Recherche année {annee}...", "year": annee, "year_progress": year_idx / total_years}

        memoires = _fetch_year(annee, mots_cles, mots_exclusion, timeout)
        yield {"type": "year_done", "annee": annee, "count": len(memoires), "year_progress": (year_idx + 0.5) / total_years}

        if extract_details:
            for i, m in enumerate(memoires):
                if not m["lien"]:
                    all_memoires.append(m)
                    continue
                titre_court = m["titre"][:60] + "…" if len(m["titre"]) > 60 else m["titre"]
                yield {
                    "type": "detail",
                    "message": f"[{i+1}/{len(memoires)}] {titre_court}",
                    "detail_progress": (i + 1) / len(memoires),
                    "year_progress": (year_idx + (i + 1) / len(memoires)) / total_years,
                }
                details = _extract_details(m["lien"], timeout)
                m.update(details)
                all_memoires.append(m)
                yield {"type": "stats", "total": len(all_memoires), "avec_pdf": sum(1 for x in all_memoires if x.get("lien_pdf")), "annee": annee}
                time.sleep(delay)
        else:
            all_memoires.extend(memoires)
            yield {"type": "stats", "total": len(all_memoires), "avec_pdf": 0, "annee": annee}

    yield {"type": "done", "total": len(all_memoires), "data": all_memoires}
