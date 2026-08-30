"""Intestazione di prompt per il testo esportato dalla Revisione.

Il TXT esportato serve quasi sempre a un solo scopo: incollarlo in una AI
(ChatGPT, Claude, DeepSeek...) e farsi tradurre o rivedere il fumetto in
blocco, con il contesto dell'intera storia che il traduttore pagina-per-pagina
della pipeline non ha. Senza istruzioni davanti, il modello riformatta a
piacere - rinumera i balloon, toglie i marcatori di pagina, aggiunge commenti -
e il file non e' piu' reimportabile.

Il prompt sta PRIMA del primo marcatore `=== pagina ===`: l'import ignora
tutto cio' che precede la prima pagina (vedi _import_edit_text), quindi il
file resta reimportabile anche senza cancellare l'intestazione a mano, e la
risposta della AI - che il prompt stesso chiede di ricominciare da `=== ... ===`
- si reimporta cosi' com'e'.
"""

import json
from pathlib import Path

# Le lingue le scrive un umano nel prompt, non un parser: nomi estesi.
_LANG_NAMES = {
    "it": "italiano",
    "en": "inglese",
    "fr": "francese",
    "es": "spagnolo",
    "de": "tedesco",
    "pt": "portoghese",
    "ja": "giapponese",
}


def _lang_name(code: str) -> str:
    code = (code or "").strip().lower()
    return _LANG_NAMES.get(code, code or "italiano")


def _glossary_lines(base: Path) -> list:
    """Le voci del glossario del fumetto, se ce n'e' uno."""
    try:
        path = Path(base) / "glossary.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            gloss = json.load(f)
        if not isinstance(gloss, dict):
            return []
    except Exception:
        return []

    righe = []
    termini = gloss.get("termini") or {}
    if isinstance(termini, dict) and termini:
        elenco = "; ".join(f"{k} -> {v}" for k, v in list(termini.items())[:60])
        righe.append(f"- Rese fisse da rispettare alla lettera: {elenco}")
    nota = (gloss.get("note") or "").strip()
    if nota:
        righe.append(f"- Contesto di questo fumetto: {nota}")
    vietate = gloss.get("vietate") or []
    if isinstance(vietate, list) and vietate:
        righe.append("- Da non usare mai: " + ", ".join(str(v) for v in vietate[:60]))
    return righe


def build(mode: str, comic: str = "", base=None,
          source_lang: str = "en", target_lang: str = "it") -> str:
    """Il testo da mettere in cima all'export.

    `mode` e' quello dell'export: "both" (originale + tradotto), "orig"
    (solo originale) o "trans" (solo tradotto). Cambia il compito: nei primi
    due la AI traduce, nel terzo rilegge una traduzione gia' fatta.
    """
    src = _lang_name(source_lang)
    dst = _lang_name(target_lang)
    titolo = f' "{comic}"' if comic else ""

    righe = [
        "### ISTRUZIONI PER LA AI (non fanno parte del fumetto) ###",
        "",
    ]

    if mode == "trans":
        righe += [
            f"Qui sotto trovi il testo tradotto in {dst} di un fumetto{titolo}, "
            "balloon per balloon, in ordine di lettura.",
            f"Rileggi l'intera traduzione: correggi errori di {dst}, refusi, "
            "incoerenze fra pagine (nomi, genere dei personaggi, tu/lei, soprannomi) "
            "e battute che non suonano naturali ad alta voce.",
        ]
    elif mode == "orig":
        righe += [
            f"Qui sotto trovi il testo originale in {src} di un fumetto{titolo}, "
            "balloon per balloon, in ordine di lettura.",
            f"Traducilo in {dst}.",
        ]
    else:
        righe += [
            f"Qui sotto trovi il testo di un fumetto{titolo}, balloon per balloon, "
            "in ordine di lettura: per ogni balloon c'e' la riga ORIGINALE "
            f"(in {src}) e la riga TRADOTTO (in {dst}).",
            f"Rivedi e correggi la traduzione in {dst} usando l'originale come "
            "riferimento e l'intera storia come contesto. Dove il TRADOTTO e' "
            "vuoto o sbagliato, riscrivilo da capo.",
        ]

    righe += [
        "",
        "Regole di traduzione:",
        "- Mantieni tono, registro, parolacce e giochi di parole del fumetto; "
        "e' dialogo parlato, non prosa: frasi brevi e naturali.",
        "- Le didascalie narrative restano didascalie, le urla restano urla; "
        "onomatopee e suoni si lasciano invariati se in italiano funzionano cosi'.",
        "- Il testo va dentro un balloon disegnato: a parita' di resa scegli la "
        "versione piu' corta, evita di allungare la battuta.",
        "- Tieni coerenti fra tutte le pagine i nomi dei personaggi, il loro genere "
        "e il modo in cui si danno del tu o del lei.",
    ]

    gl = _glossary_lines(base) if base is not None else []
    if gl:
        righe += ["", "Glossario del fumetto (ha la precedenza su tutto il resto):"] + gl

    righe += [
        "",
        "Regole di formato (il file viene reimportato da un programma, "
        "se cambi il formato si perde tutto):",
        # Niente marcatori di pagina letterali qui dentro: il parser di import
        # li prenderebbe per veri e inizierebbe una pagina inesistente.
        "- Rispondi con il solo testo nello stesso identico formato: la riga con "
        "il nome della pagina fra tripli uguali, poi le righe dei balloon.",
        "- Non cambiare, non riordinare e non rinumerare i codici fra parentesi "
        "quadre: sono gli identificativi dei balloon.",
        "- Non saltare nessun balloon, nemmeno quelli vuoti o gia' corretti: "
        "riportali comunque.",
        # Il file contiene righe diverse a seconda della modalita': la regola
        # deve parlare solo di quelle che ci sono davvero, altrimenti chiede
        # di conservare righe ORIGINALE che nell'export non esistono.
        ("- Per ogni balloon riporta la riga ORIGINALE invariata e aggiungi sotto "
         "la riga TRADOTTO con lo stesso codice, nella forma "
         "[codice] TRADOTTO: testo" if mode == "orig" else
         "- Riscrivi ogni riga TRADOTTO, corretta dove serve, con lo stesso codice."
         if mode == "trans" else
         "- Riporta le righe ORIGINALE invariate; modifica solo le righe TRADOTTO."),
        "- Ogni balloon sta su una riga sola: niente a capo dentro il testo.",
        "- Nessun commento, nessuna spiegazione, nessun blocco di codice attorno.",
        "",
        "### FINE ISTRUZIONI - da qui inizia il fumetto ###",
        "",
    ]
    return "\n".join(righe)


def langs_from_cfg(cfg: dict):
    """(source_lang, target_lang) del traduttore attivo nel config.

    Le lingue stanno nella sezione `local_llm:`, la stessa che usa
    translate_local.lang_pair().
    """
    sezione = cfg.get("local_llm") or {}
    if not isinstance(sezione, dict):
        sezione = {}
    return sezione.get("source_lang", "en"), sezione.get("target_lang", "it")
