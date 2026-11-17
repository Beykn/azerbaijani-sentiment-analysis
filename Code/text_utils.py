# -*- coding: utf-8 -*-
"""
Azerbaijani text utilities shared by every script in this project.

Usage:
    from text_utils import normalize_text_az, detect_domain, to_class_label

Placeholder tokens produced by the cleaner (always UPPERCASE so they can never
collide with real words, which are all lowercase after cleaning):
    URL, EMAIL, PHONE, USER, EMO_POS, EMO_NEG, <NUM>,
    <PRICE>, <STARS_1>..<STARS_5>, <RATING_POS>, <RATING_NEG>

NOTE for Keras: the default keras Tokenizer strips "_", "<" and ">", which
would break tokens like "yaxşı_NEG" or "<NUM>". Use it on already-cleaned
text like this:
    Tokenizer(filters="", lower=False, split=" ", oov_token="<OOV>")
"""

import html
import re
import unicodedata

try:
    from ftfy import fix_text  # fixes broken encodings, e.g. "ÃƒÂ§" -> "ç"
except ImportError:  # ftfy is optional
    def fix_text(s: str) -> str:
        return s


# ---------------------------------------------------------------------------
# 1. Azerbaijani-aware lowercase
# ---------------------------------------------------------------------------
def lower_az(s: str) -> str:
    """Lowercase with Azerbaijani rules: I -> ı, İ -> i (Python's lower() gets these wrong)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("I", "ı").replace("İ", "i")
    s = s.lower().replace("i̇", "i")  # "i" + combining dot left over by some inputs
    return s


# ---------------------------------------------------------------------------
# 2. Patterns and small dictionaries
# ---------------------------------------------------------------------------
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_RE = re.compile(r"([a-zəğıöüçş])([A-ZƏĞİÖÜÇŞ])")
MULTI_PUNCT_RE = re.compile(r"([!?.,;:])\1+")
MULTI_SPACE_RE = re.compile(r"\s+")
REPEAT_CHARS_RE = re.compile(r"(.)\1{2,}")  # cooool -> cool

# Characters to keep after cleaning: letters/digits/underscore (\w), spaces,
# "<" ">" for placeholders, apostrophe for words like "o'n".
# keep_sentence_punct=True also keeps . ! ? so sentences can be split later.
STRIP_RE = re.compile(r"[^\w\s<>']")
STRIP_KEEP_SENT_RE = re.compile(r"[^\w\s<>'.!?]")

AZ_LETTERS = "A-Za-zƏəĞğIıİiÖöÜüÇçŞşXxQq"
# Placeholders are listed FIRST so "EMO_POS" is not cut into "EMO" + "POS".
TOKEN_RE = re.compile(
    r"<[A-Z_0-9]+>"
    r"|EMO_(?:POS|NEG)|URL|EMAIL|PHONE|USER"
    rf"|[{AZ_LETTERS}]+(?:'[{AZ_LETTERS}]+)?"
)

EMO_MAP = {
    "🙂": "EMO_POS", "😀": "EMO_POS", "😁": "EMO_POS", "😂": "EMO_POS", "😍": "EMO_POS",
    "😊": "EMO_POS", "👍": "EMO_POS", "❤": "EMO_POS", "🥰": "EMO_POS", "👏": "EMO_POS",
    "☹": "EMO_NEG", "🙁": "EMO_NEG", "😠": "EMO_NEG", "😡": "EMO_NEG", "👎": "EMO_NEG",
    "😢": "EMO_NEG", "😭": "EMO_NEG", "🤬": "EMO_NEG", "💔": "EMO_NEG",
}

SLANG_MAP = {"slm": "salam", "tmm": "tamam", "sagol": "sağol", "cox": "çox", "yaxsi": "yaxşı"}

NEGATORS = {"yox", "deyil", "heç", "qətiyyən", "yoxdur"}
NEG_SCOPE = 3  # how many tokens after a negator get the "_NEG" suffix

STOPWORDS = {
    "və", "ilə", "amma", "ancaq", "lakin", "ya", "həm", "ki", "bu", "bir",
    "biz", "siz", "mən", "sən", "orada", "burada", "bütün", "hər", "artıq",
    "az", "ən", "də", "da", "üçün",
}  # negators are intentionally NOT here


# ---------------------------------------------------------------------------
# 3. Domain detection + domain-specific normalization
# ---------------------------------------------------------------------------
NEWS_HINTS = re.compile(r"\b(apa|trend|azertac|reuters|bloomberg|dha|aa)\b", re.I)
SOCIAL_HINTS = re.compile(r"\b(rt)\b|@|#|(?:😂|😍|😊|👍|👎|😡|🙂)")
REV_HINTS = re.compile(r"\b(azn|manat|qiymət|aldım|ulduz|çox yaxşı|çox pis)\b", re.I)

# These run on lowercased text BEFORE digits become <NUM>,
# otherwise "100 manat" would already be "<NUM> manat" and never match.
PRICE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:azn|manat)\b")
STARS_RE = re.compile(r"\b([1-5])\s*ulduz\b")
POS_RATE_RE = re.compile(r"\b(?:çox|cox) (?:yaxşı|yaxsi)\b")
NEG_RATE_RE = re.compile(r"\b(?:çox|cox) pis\b")


def detect_domain(text: str) -> str:
    """Very simple rule-based domain guess: news / social / reviews / general."""
    s = lower_az(str(text))
    if NEWS_HINTS.search(s):
        return "news"
    if SOCIAL_HINTS.search(s):
        return "social"
    if REV_HINTS.search(s):
        return "reviews"
    return "general"


def _review_normalize(s: str) -> str:
    s = PRICE_RE.sub(" <PRICE> ", s)
    s = STARS_RE.sub(lambda m: f" <STARS_{m.group(1)}> ", s)
    s = POS_RATE_RE.sub(" <RATING_POS> ", s)
    s = NEG_RATE_RE.sub(" <RATING_NEG> ", s)
    return s


def add_domain_tag(line: str, domain: str) -> str:
    """'film əla idi', 'reviews' -> 'domreviews film əla idi'"""
    return f"dom{domain} {line}"


# ---------------------------------------------------------------------------
# 4. Main cleaner
# ---------------------------------------------------------------------------
def _is_placeholder(tok: str) -> bool:
    # Real words are lowercase after cleaning, placeholders are uppercase.
    return tok.startswith("<") or tok.isupper()


def normalize_text_az(
    s,
    domain: str | None = None,
    numbers_to_token: bool = True,
    keep_sentence_punct: bool = False,
    mark_negation: bool = True,
    remove_stopwords: bool = False,
) -> str:
    """
    Clean one Azerbaijani text and return space-separated tokens.

    domain: pass "reviews" to get <PRICE>/<STARS_n>/<RATING_*> tokens.
            Pass None to skip domain-specific rules.
    """
    if not isinstance(s, str):
        return ""

    # --- a) repair and strip markup -------------------------------------
    s = fix_text(s)
    s = html.unescape(s)
    s = HTML_TAG_RE.sub(" ", s)

    # --- b) things that need the ORIGINAL casing ------------------------
    s = URL_RE.sub(" \x00URL\x00 ", s)        # URLs may contain "#" and "@"
    s = EMAIL_RE.sub(" \x00EMAIL\x00 ", s)    # emails contain "@"
    # #QarabagIsBack -> "qarabag is back" (needs capitals to find word borders).
    # Plain lower() here: hashtags are usually Latin/English, and lower_az
    # would turn "Is" into "ıs".
    s = HASHTAG_RE.sub(
        lambda m: " " + CAMEL_RE.sub(r"\1 \2", m.group(1)).lower() + " ", s
    )

    # --- c) lowercase, then insert UPPERCASE placeholders ----------------
    # Placeholders are added AFTER lower_az(); otherwise its I -> ı rule
    # would turn "EMAIL" into "emaıl".
    s = lower_az(s)
    s = s.replace("\x00url\x00", " URL ").replace("\x00emaıl\x00", " EMAIL ")
    for emo, tag in EMO_MAP.items():
        s = s.replace(emo, f" {tag} ")
    s = PHONE_RE.sub(" PHONE ", s)
    s = USER_RE.sub(" USER ", s)

    if domain == "reviews":
        s = _review_normalize(s)

    s = MULTI_PUNCT_RE.sub(r"\1", s)
    if numbers_to_token:
        # (?<!_) keeps the digit inside "<STARS_5>" untouched
        s = re.sub(r"(?<!_)\d+", " <NUM> ", s)

    # --- d) drop punctuation -------------------------------------------
    s = (STRIP_KEEP_SENT_RE if keep_sentence_punct else STRIP_RE).sub(" ", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    if keep_sentence_punct:
        return s  # caller splits on . ! ? and cleans each part again

    # --- e) token-level rules ------------------------------------------
    out = []
    neg_left = 0
    for t in TOKEN_RE.findall(s):
        if not _is_placeholder(t):
            t = REPEAT_CHARS_RE.sub(r"\1\1", t)
            t = SLANG_MAP.get(t, t)

        if len(t) == 1 and t not in {"o", "e"}:
            continue  # single letters carry no meaning
        if remove_stopwords and t in STOPWORDS:
            continue

        if mark_negation and t in NEGATORS:
            out.append(t)
            neg_left = NEG_SCOPE
            continue
        if neg_left > 0 and not _is_placeholder(t):
            out.append(t + "_NEG")
            neg_left -= 1
        else:
            out.append(t)

    return " ".join(out)


def tokenize_az(s, **kwargs) -> list[str]:
    """Same as normalize_text_az but returns a token list (handy for gensim)."""
    return normalize_text_az(s, **kwargs).split()


# ---------------------------------------------------------------------------
# 5. Labels
# ---------------------------------------------------------------------------
_POS = {"pos", "positive", "müsbət", "pozitiv", "good"}
_NEU = {"neu", "neutral", "neytral"}
_NEG = {"neg", "negative", "mənfi", "neqativ", "bad"}


def to_class_label(v, scheme: str):
    """
    Map any raw label to a class id: 0 = Negative, 1 = Neutral, 2 = Positive.

    scheme="binary": 0 -> 0, 1 -> 2   (binary sets have no neutral class)
    scheme="tri":    text labels (positive/neutral/negative, any case)
    Returns None for unknown labels so the row can be dropped.
    """
    if scheme == "binary":
        try:
            return 2 if int(v) == 1 else 0
        except (TypeError, ValueError):
            return None
    s = str(v).strip().lower()
    if s in _POS:
        return 2
    if s in _NEU:
        return 1
    if s in _NEG:
        return 0
    return None


def class_to_value(c: int) -> float:
    """0/1/2 class id -> Part-1 style score 0.0/0.5/1.0."""
    return {0: 0.0, 1: 0.5, 2: 1.0}[int(c)]


# ---------------------------------------------------------------------------
# Quick self-test:  python text_utils.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        ("İLHAM ƏLİYEV BAKIDA", None),
        ("Bax www.example.az və yaz info@mail.az 😍 @ali", None),
        ("#QarabagIsBack cooool!!!", None),
        ("Bu film yaxşı deyil, heç xoşuma gəlmədi", None),
        ("100 manat verdim, 5 ulduz, çox yaxşı məhsul", "reviews"),
        ("slm, cox yaxsi idi 👍", None),
    ]
    for text, dom in examples:
        print(f"{text!r}\n  -> {normalize_text_az(text, domain=dom)!r}")
    print("domain:", detect_domain("100 AZN qiymət"), detect_domain("RT @user salam"))
    print("labels:", to_class_label("Positive", "tri"), to_class_label(0, "binary"),
          to_class_label("neutral", "tri"), to_class_label(1, "binary"))
