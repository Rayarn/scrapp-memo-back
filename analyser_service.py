import json
from openai import OpenAI
from datetime import datetime
from typing import Generator


def analyse_subject(corpus: list[dict], params: dict, api_key: str, model: str) -> dict:
    """Analyze user's thesis project against the corpus. Returns structured result."""
    MAX_CHARS = 90_000
    lines = []
    for row in corpus:
        titre = str(row.get("titre", ""))
        auteur = str(row.get("auteur", ""))
        annee = str(row.get("annee", ""))
        resume = str(row.get("resume_fr", ""))
        if resume.strip() in ("", "nan", "None"):
            resume = ""
        entry = f"[{annee}] {titre} — {auteur}"
        if resume:
            entry += f"\nRésumé : {resume[:500]}"
        lines.append(entry)

    corpus_text = "\n\n".join(lines)
    if len(corpus_text) > MAX_CHARS:
        corpus_text = corpus_text[:MAX_CHARS] + "\n\n[... corpus tronqué ...]"

    mots_cles_str = ", ".join(params.get("mots_cles", [])) or "non précisés"
    sujet_titre = params.get("titre", "")
    sujet_description = params.get("description", "")
    sujet_angle = params.get("angle", "")

    prompt = f"""Tu es un expert en actuariat et en recherche académique.

Un étudiant actuaire prépare son mémoire. Voici son projet :

---
TITRE PROVISOIRE : {sujet_titre or "(non précisé)"}
DESCRIPTION : {sujet_description}
MOTS-CLÉS : {mots_cles_str}
ANGLE PRINCIPAL : {sujet_angle}
---

Voici le corpus complet des mémoires déjà soutenus à l'Institut des Actuaires (titres + résumés) :

{corpus_text}

---

En te basant UNIQUEMENT sur ce corpus, réalise une analyse structurée en 5 parties.
Ne suppose aucun domaine a priori : raisonne exclusivement à partir de ce que l'étudiant a décrit et de ce que contient le corpus.

## 1. 🔴 Mémoires quasi-identiques (risque fort de redite)
Liste les mémoires du corpus dont le sujet, la méthode ou la problématique est très proche du projet.
Pour chacun : [Année] Titre — Auteur | En quoi il se recoupe avec le projet.
Si aucun : dis-le explicitement.

## 2. 🟡 Mémoires connexes (même domaine ou méthode, angle différent)
Liste les mémoires qui partagent un thème, une méthode ou une population avec le projet, sans être identiques.
Explique brièvement la différence avec le projet de l'étudiant.

## 3. 🟢 Ce qui est ORIGINAL dans le projet
Identifie précisément ce que l'étudiant apporterait de nouveau par rapport au corpus.

## 4. 💡 Recommandations pour renforcer l'originalité
Propose 3 à 5 ajustements concrets au projet.

## 5. 📚 Mémoires du corpus à lire en priorité
Liste 5 à 8 mémoires. Format : [Année] Titre — Auteur | Pourquoi le lire

Sois précis et factuel. Réponds en français."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=3000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Tu es un expert en actuariat et en recherche académique. Tu réponds en français avec un style académique clair en utilisant du markdown."},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    return {
        "text": text,
        "titre": sujet_titre,
        "description": sujet_description,
        "mots_cles": mots_cles_str,
        "angle": sujet_angle,
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "nb_memoires": len(corpus),
    }


def classify_memoire(titre: str, resume: str, api_key: str, model: str, max_tokens: int = 500, temperature: float = 0.0) -> dict:
    """Classify a single memoir. Returns structured JSON."""
    if not resume or not resume.strip():
        return {"pertinence": 0, "themes": [], "methodes": [], "mots_cles_extraits": [], "synthese": "", "erreur": "résumé vide"}
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Tu es un expert en actuariat. Tu réponds UNIQUEMENT en JSON valide."},
                {"role": "user", "content": f"""Analyse ce résumé de mémoire d'actuariat.

TITRE : {titre}
RÉSUMÉ : {resume}

Réponds avec exactement ce JSON :
{{
  "pertinence": <entier 0-100>,
  "themes": [<thèmes abordés>],
  "methodes": [<méthodes/modèles mentionnés>],
  "mots_cles_extraits": [<3 à 6 mots-clés>],
  "synthese": "<1-2 phrases résumant la contribution>"
}}

Ne suppose aucun domaine a priori. Déduis tout du contenu."""},
            ],
        )
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError as e:
        return {"pertinence": 0, "themes": [], "methodes": [], "mots_cles_extraits": [], "synthese": "", "erreur": f"JSON: {e}"}
    except Exception as e:
        return {"pertinence": 0, "themes": [], "methodes": [], "mots_cles_extraits": [], "synthese": "", "erreur": str(e)}


def classify_corpus(corpus: list[dict], api_key: str, model: str, max_tokens: int, temperature: float, min_resume_length: int) -> Generator[dict, None, None]:
    """Generator yielding classification progress events."""
    to_analyse = [r for r in corpus if len(str(r.get("resume_fr", ""))) >= min_resume_length]
    total = len(to_analyse)
    results = []
    cost_per = 0.001 if "mini" in model else (0.010 if "turbo" in model else 0.005)
    total_cost = 0.0

    for i, row in enumerate(to_analyse):
        titre = str(row.get("titre", ""))
        resume = str(row.get("resume_fr", ""))
        yield {"type": "progress", "current": i + 1, "total": total, "titre": titre[:70]}

        result = classify_memoire(titre, resume, api_key, model, max_tokens, temperature)
        total_cost += cost_per

        enriched = {**row}
        enriched["pertinence"] = result.get("pertinence", 0)
        enriched["themes"] = ", ".join(result.get("themes", []))
        enriched["methodes"] = ", ".join(result.get("methodes", []))
        enriched["mots_cles_extraits"] = ", ".join(result.get("mots_cles_extraits", []))
        enriched["synthese"] = result.get("synthese", "")
        if "erreur" in result:
            enriched["erreur"] = result["erreur"]
        results.append(enriched)

        scores = [r["pertinence"] for r in results if isinstance(r.get("pertinence"), (int, float))]
        yield {
            "type": "stats",
            "analysed": len(results),
            "pertinents": sum(1 for s in scores if s >= 50),
            "score_moyen": round(sum(scores) / len(scores)) if scores else 0,
            "cout": round(total_cost, 3),
        }

    yield {"type": "done", "data": results, "cout_total": round(total_cost, 3)}
