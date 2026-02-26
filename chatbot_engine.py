"""
AirBot Production ML Engine — v2.0
====================================
100% Free. No external API. No API key needed.

ML Techniques used:
- TF-IDF Vectorization (search engine algorithm)
- Cosine Similarity (intent matching)
- Fuzzy String Matching (typo correction)
- Multi-intent Detection (handles complex questions)
- Confidence Scoring (knows when it's unsure)
- Entity Extraction (city, state, pollutant, number)
"""

import re
import math
import pandas as pd
from difflib import get_close_matches, SequenceMatcher

# ════════════════════════════════════════════
# 1. LOAD & PREPARE DATASET
# ════════════════════════════════════════════
df = pd.read_csv("air_quality.csv")
df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
df["pollutant_min"] = pd.to_numeric(df["pollutant_min"], errors="coerce")
df["pollutant_max"] = pd.to_numeric(df["pollutant_max"], errors="coerce")

# Build lookup dictionaries (lowercase → original case)
CITY_DISPLAY  = {c.lower(): c for c in df["city"].unique()}
STATE_DISPLAY = {s.lower(): s for s in df["state"].unique()}

ALL_CITIES     = sorted(CITY_DISPLAY.keys())
ALL_STATES     = sorted(STATE_DISPLAY.keys())
ALL_POLLUTANTS = sorted(df["pollutant_id"].str.upper().unique().tolist())

# State name aliases (underscores, spaces, abbreviations)
STATE_ALIASES = {
    "andhra pradesh": "andhra_pradesh",
    "ap": "andhra_pradesh",
    "arunachal pradesh": "arunachal_pradesh",
    "arunachal": "arunachal_pradesh",
    "j&k": "jammu_and_kashmir",
    "jammu kashmir": "jammu_and_kashmir",
    "jammu and kashmir": "jammu_and_kashmir",
    "jk": "jammu_and_kashmir",
    "mp": "madhya pradesh",
    "madhya pradesh": "madhya pradesh",
    "up": "uttar_pradesh",
    "uttar pradesh": "uttar_pradesh",
    "wb": "west_bengal",
    "west bengal": "west_bengal",
    "hp": "himachal pradesh",
    "himachal": "himachal pradesh",
    "himachal pradesh": "himachal pradesh",
    "tn": "tamilnadu",
    "tamil nadu": "tamilnadu",
    "tamilnadu": "tamilnadu",
    "uk": "uttarakhand",
    "uttaranchal": "uttarakhand",
    "pb": "punjab",
    "rj": "rajasthan",
    "raj": "rajasthan",
    "mh": "maharashtra",
    "ts": "telangana",
}

# City aliases
CITY_ALIASES = {
    "del": "delhi", "dilli": "delhi", "dehli": "delhi", "new delhi": "delhi",
    "mum": "mumbai", "bombay": "mumbai", "mumabi": "mumbai", "mumbay": "mumbai",
    "blore": "bengaluru", "bangalore": "bengaluru", "blr": "bengaluru", "bangaluru": "bengaluru",
    "hyd": "hyderabad", "hydrabad": "hyderabad", "hyderbad": "hyderabad",
    "kolkatta": "kolkata", "calcutta": "kolkata",
    "chenai": "chennai", "madras": "chennai",
    "puna": "pune", "poona": "pune",
    "ahmedabad": "ahmedabad", "ahemdabad": "ahmedabad",
    "bhopal": "bhopal",
    "lucknow": "lucknow", "lko": "lucknow",
    "jaipur": "jaipur", "jpj": "jaipur",
    "chandigarh": "chandigarh", "chd": "chandigarh",
    "patna": "patna",
    "bhubaneswar": "bhubaneswar",
    "guwahati": "guwahati",
    "srinagar": "srinagar",
    "amritsar": "amritsar",
}

# ════════════════════════════════════════════
# 2. SPELLING & ENTITY CORRECTION
# ════════════════════════════════════════════
WORD_CORRECTIONS = {
    "avrage": "average", "averge": "average", "averg": "average", "avg": "average",
    "pollustion": "pollution", "pollusion": "pollution", "polution": "pollution",
    "emmision": "emission", "emision": "emission", "emision": "emission",
    "comparision": "comparison", "comparsion": "comparison",
    "higest": "highest", "highst": "highest", "hiest": "highest",
    "lowset": "lowest", "lowst": "lowest", "lwest": "lowest",
    "maximun": "maximum", "maxmum": "maximum", "maxium": "maximum",
    "minimun": "minimum", "minmum": "minimum", "minium": "minimum",
    "safest": "safest", "cleanest": "cleanest",
    "pm25": "pm2.5", "pm 25": "pm2.5", "pm2 5": "pm2.5",
    "pm 10": "pm10",
    "ozon": "ozone", "ozne": "ozone", "ozonegas": "ozone",
    "nitogen": "no2", "nitrogen dioxide": "no2", "nitrogen": "no2",
    "sulphur": "so2", "sulfur": "so2", "sulphur dioxide": "so2", "sulfur dioxide": "so2",
    "carbon monoxide": "co",
    "ammonia": "nh3",
    "particulate": "pm2.5",
    "dangereous": "dangerous", "dangrous": "dangerous",
    "helath": "health", "healt": "health",
    "comapr": "compare", "compar": "compare",
    "betwen": "between", "betwn": "between",
    "citeis": "cities", "citys": "cities",
    "staes": "states", "sates": "states", "satte": "state",
    "wrost": "worst", "wrst": "worst",
    "wat": "what", "wht": "what", "whats": "what is",
    "hw": "how", "hwo": "how",
    "mny": "many", "meny": "many",
    "whcih": "which", "wich": "which",
    "levl": "level", "lvl": "level", "levle": "level",
}

def correct_spelling(text):
    """Fix common typos word by word."""
    text = text.lower().strip()
    words = text.split()
    corrected = []
    i = 0
    while i < len(words):
        # Try two-word phrases first
        if i + 1 < len(words):
            two = words[i] + " " + words[i+1]
            if two in WORD_CORRECTIONS:
                corrected.append(WORD_CORRECTIONS[two])
                i += 2
                continue
        word = words[i]
        corrected.append(WORD_CORRECTIONS.get(word, word))
        i += 1
    return " ".join(corrected)

def fuzzy_match(word, candidates, cutoff=0.75):
    """Find closest match using SequenceMatcher (handles typos)."""
    best_match = None
    best_ratio = 0
    for candidate in candidates:
        ratio = SequenceMatcher(None, word.lower(), candidate.lower()).ratio()
        if ratio > best_ratio and ratio >= cutoff:
            best_ratio = ratio
            best_match = candidate
    return best_match, best_ratio

def find_all_cities(text):
    """Extract ALL cities mentioned in text."""
    text_lower = text.lower()
    found = []

    # Check aliases first
    for alias, city in CITY_ALIASES.items():
        if alias in text_lower and city in CITY_DISPLAY:
            display = CITY_DISPLAY[city]
            if display not in found:
                found.append(display)

    # Direct match
    for city in ALL_CITIES:
        if city in text_lower:
            display = CITY_DISPLAY[city]
            if display not in found:
                found.append(display)

    # Fuzzy match on individual words if nothing found yet
    if not found:
        words = re.findall(r'\b\w{4,}\b', text_lower)
        for word in words:
            match, ratio = fuzzy_match(word, ALL_CITIES, cutoff=0.80)
            if match:
                display = CITY_DISPLAY[match]
                if display not in found:
                    found.append(display)

    return found

def find_all_states(text):
    """Extract ALL states mentioned in text."""
    text_lower = text.lower()
    found = []

    # Check aliases
    for alias, state_key in STATE_ALIASES.items():
        if alias in text_lower:
            # Find the display name
            for s_lower, s_display in STATE_DISPLAY.items():
                if state_key in s_lower or s_lower in state_key:
                    if s_display not in found:
                        found.append(s_display)

    # Direct match (handle underscores and spaces)
    for state_lower, state_display in STATE_DISPLAY.items():
        state_normalized = state_lower.replace("_", " ")
        if state_normalized in text_lower or state_lower in text_lower:
            if state_display not in found:
                found.append(state_display)

    # Fuzzy match
    if not found:
        words = re.findall(r'\b\w{4,}\b', text_lower)
        for word in words:
            match, ratio = fuzzy_match(word, ALL_STATES, cutoff=0.80)
            if match:
                display = STATE_DISPLAY[match]
                if display not in found:
                    found.append(display)

    return found

def find_all_pollutants(text):
    """Extract ALL pollutants mentioned in text."""
    text_upper = text.upper()
    found = []
    # Order matters — PM2.5 before PM
    for pol in ["PM2.5", "PM10", "NO2", "SO2", "NH3", "OZONE", "CO"]:
        if pol in text_upper:
            found.append(pol)
    # Natural language
    if not found or "PM2.5" not in found:
        if "FINE PARTICLE" in text_upper or "FINE PARTICULATE" in text_upper:
            found.append("PM2.5")
    if "NITROGEN" in text_upper and "NO2" not in found:
        found.append("NO2")
    if ("SULPHUR" in text_upper or "SULFUR" in text_upper) and "SO2" not in found:
        found.append("SO2")
    if "AMMONIA" in text_upper and "NH3" not in found:
        found.append("NH3")
    if "CARBON MONO" in text_upper and "CO" not in found:
        found.append("CO")
    if ("OZON" in text_upper or "O3" in text_upper) and "OZONE" not in found:
        found.append("OZONE")
    return found

def extract_number(text):
    """Extract a number from text (e.g. 'top 3 cities' → 3)."""
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for word, num in words.items():
        if word in text.lower():
            return num
    nums = re.findall(r'\b(\d+)\b', text)
    if nums:
        return int(nums[0])
    return 5  # default

# ════════════════════════════════════════════
# 3. TF-IDF + COSINE SIMILARITY ENGINE
# ════════════════════════════════════════════
INTENT_PATTERNS = {
    "average": [
        "average level", "mean value", "avg reading", "what is the level",
        "how much pollution", "pollution level", "reading in", "value in",
        "concentration in", "measure in", "level of pollutant",
    ],
    "top_polluted": [
        "highest pollution", "most polluted", "worst air", "top cities pollution",
        "maximum pollution", "cities with most", "rank highest", "most affected",
        "which city has highest", "worst city", "most dangerous city",
    ],
    "least_polluted": [
        "lowest pollution", "least polluted", "cleanest city", "best air quality",
        "safest city", "minimum pollution", "cities with least", "rank lowest",
        "which city has lowest", "cleanest air", "good air quality city",
    ],
    "compare": [
        "compare cities", "versus", "city vs city", "which is better",
        "difference between", "side by side", "both cities", "two cities",
        "compare states", "state vs state", "which state is worse",
    ],
    "state_ranking": [
        "worst state", "best state", "most polluted state", "cleanest state",
        "rank states", "state with worst", "state with best", "state comparison",
        "which state has", "top states pollution",
    ],
    "city_report": [
        "all pollutants", "full report", "complete data", "all data for",
        "everything about", "all readings", "show all", "full details",
        "all levels in", "complete air quality",
    ],
    "state_report": [
        "all cities in state", "state overview", "all data for state",
        "state air quality", "pollution in state", "state full report",
    ],
    "health_check": [
        "is it safe", "is air safe", "safe to breathe", "health risk",
        "dangerous level", "harmful", "should i worry", "health impact",
        "is it dangerous", "pollution safe", "livable",
    ],
    "explain_pollutant": [
        "what is pollutant", "explain pollutant", "tell me about pollutant",
        "what does mean", "describe pollutant", "health effects",
        "difference between pm", "pm25 vs pm10", "what causes",
    ],
    "cities_in_state": [
        "cities in state", "which cities in", "list cities", "how many cities in",
        "cities monitored in", "stations in state", "coverage in",
    ],
    "national_average": [
        "india average", "national average", "country average", "overall average",
        "across india", "india wide", "pan india", "nationwide average",
    ],
    "safe_cities": [
        "safe cities", "cities below limit", "who limit", "within safe",
        "cities with good air", "safe air quality cities", "pass who standard",
    ],
    "greeting": [
        "hello", "hi there", "hey", "good morning", "good evening",
        "what can you do", "help me", "how can you help", "what do you know",
        "capabilities", "features",
    ],
    "dataset_info": [
        "how many records", "total data", "dataset size", "data coverage",
        "how many cities", "how many states", "what data available",
        "which cities covered", "what is your data",
    ],
}

def tokenize(text):
    text = re.sub(r"[^\w\s.]", " ", text.lower())
    return [w for w in text.split() if len(w) > 1]

def build_tfidf():
    """Build TF-IDF model from intent patterns."""
    docs, doc_intents = [], []
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            docs.append(pattern)
            doc_intents.append(intent)

    tokenized = [tokenize(d) for d in docs]
    all_words = set(w for doc in tokenized for w in doc)
    N = len(docs)

    # IDF calculation
    idf = {}
    for word in all_words:
        count = sum(1 for doc in tokenized if word in doc)
        idf[word] = math.log((N + 1) / (count + 1)) + 1

    # TF-IDF vectors
    vectors = []
    for tokens in tokenized:
        vec = {}
        for word in tokens:
            tf = tokens.count(word) / len(tokens)
            vec[word] = tf * idf.get(word, 0)
        vectors.append(vec)

    return vectors, doc_intents, idf

TFIDF_VECS, DOC_INTENTS, IDF = build_tfidf()

def cosine_sim(v1, v2):
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[w] * v2[w] for w in common)
    m1 = math.sqrt(sum(x**2 for x in v1.values()))
    m2 = math.sqrt(sum(x**2 for x in v2.values()))
    return dot / (m1 * m2) if m1 and m2 else 0.0

def classify_intent(text, confidence_threshold=0.15):
    """Classify intent using TF-IDF + Cosine Similarity with confidence score."""
    q_vec = {}
    tokens = tokenize(text)
    for word in tokens:
        tf = tokens.count(word) / len(tokens)
        q_vec[word] = tf * IDF.get(word, 0)

    scores = {}
    for vec, intent in zip(TFIDF_VECS, DOC_INTENTS):
        score = cosine_sim(q_vec, vec)
        scores[intent] = max(scores.get(intent, 0), score)

    if not scores:
        return "unknown", 0.0
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    return (best_intent, best_score) if best_score >= confidence_threshold else ("unknown", best_score)

# ════════════════════════════════════════════
# 4. KEYWORD OVERRIDE (for high accuracy)
# ════════════════════════════════════════════
def keyword_intent(text):
    """
    Fast keyword matching — runs before TF-IDF.
    Catches clear questions that don't need ML.
    """
    t = text.lower()

    # Comparison patterns
    if re.search(r'\b(vs|versus|compare|comparison|difference between|and)\b', t) and \
       (len(find_all_cities(t)) >= 2 or len(find_all_states(t)) >= 2):
        return "compare"

    # Ranking patterns
    if re.search(r'\b(top|highest|most polluted|worst|maximum|max)\b', t):
        if re.search(r'\b(cit(y|ies)|place|location)\b', t) or find_all_pollutants(t):
            return "top_polluted"

    if re.search(r'\b(lowest|least|cleanest|best|minimum|min|safest)\b', t):
        if re.search(r'\b(cit(y|ies)|place|location)\b', t):
            return "least_polluted"

    # Average patterns
    if re.search(r'\b(average|avg|mean|level|reading|concentration|value|how much)\b', t):
        if find_all_cities(t) or find_all_states(t):
            if find_all_pollutants(t) or re.search(r'\b(pollution|air quality)\b', t):
                return "average"

    # Health check
    if re.search(r'\b(safe|dangerous|harmful|healthy|breathe|risk|hazard|livable)\b', t):
        return "health_check"

    # Explain
    if re.search(r'\b(what is|explain|meaning|describe|define|difference)\b', t):
        if find_all_pollutants(t):
            return "explain_pollutant"

    # State ranking
    if re.search(r'\b(worst|best|most|least)\b', t) and \
       re.search(r'\b(state|states|region|province)\b', t):
        return "state_ranking"

    # Cities in state
    if re.search(r'\b(cities|city list|stations|how many cities)\b', t) and find_all_states(t):
        return "cities_in_state"

    # Full report
    if re.search(r'\b(all|complete|full|everything|entire|report|overview)\b', t):
        if find_all_cities(t):
            return "city_report"
        if find_all_states(t):
            return "state_report"

    # National average
    if re.search(r'\b(india|national|country|overall|pan india|nationwide)\b', t):
        if find_all_pollutants(t) or re.search(r'\b(average|avg|level)\b', t):
            return "national_average"

    # Safe cities
    if re.search(r'\b(safe cities|who limit|good air|clean air cities|pass standard)\b', t):
        return "safe_cities"

    # Dataset info
    if re.search(r'\b(how many|total|count|dataset|coverage|records)\b', t):
        if re.search(r'\b(cities|states|records|data|readings)\b', t):
            return "dataset_info"

    # Greeting
    if re.search(r'\b(hello|hi|hey|help|what can you|capabilities)\b', t):
        return "greeting"

    return None

# ════════════════════════════════════════════
# 5. DATA COMPUTATION
# ════════════════════════════════════════════
POLLUTANT_INFO = {
    "PM2.5": {
        "name": "Fine Particulate Matter (PM2.5)",
        "desc": "Microscopic particles under 2.5 micrometers — penetrate deep into lungs and enter bloodstream.",
        "who_limit": 15, "unit": "μg/m³",
        "health": "Causes lung disease, heart attacks, strokes, cancer. Most dangerous urban pollutant.",
        "sources": "Vehicle exhaust, factory emissions, construction dust, burning crop residue.",
        "risk_levels": [(15, "✅ Safe"), (35, "🟡 Moderate"), (55, "🟠 Unhealthy for sensitive groups"), (150, "🔴 Unhealthy"), (999, "🟣 Hazardous")],
    },
    "PM10": {
        "name": "Coarse Particulate Matter (PM10)",
        "desc": "Particles under 10 micrometers — dust, pollen, mold spores.",
        "who_limit": 45, "unit": "μg/m³",
        "health": "Irritates respiratory tract, triggers asthma attacks, aggravates allergies.",
        "sources": "Road dust, construction sites, mining operations, agricultural activities.",
        "risk_levels": [(45, "✅ Safe"), (100, "🟡 Moderate"), (250, "🔴 Unhealthy"), (999, "🟣 Hazardous")],
    },
    "NO2": {
        "name": "Nitrogen Dioxide (NO2)",
        "desc": "Reddish-brown gas produced by combustion. Key traffic pollutant.",
        "who_limit": 10, "unit": "μg/m³",
        "health": "Inflames airways, increases asthma attacks, reduces lung function over time.",
        "sources": "Vehicle engines (especially diesel), thermal power plants, industrial boilers.",
        "risk_levels": [(10, "✅ Safe"), (40, "🟡 Moderate"), (200, "🔴 Unhealthy"), (999, "🟣 Hazardous")],
    },
    "SO2": {
        "name": "Sulfur Dioxide (SO2)",
        "desc": "Colorless gas with pungent smell. Acid rain precursor.",
        "who_limit": 40, "unit": "μg/m³",
        "health": "Causes respiratory illness, aggravates asthma, eye and skin irritation.",
        "sources": "Coal-burning power plants, oil refineries, metal smelters, volcanoes.",
        "risk_levels": [(40, "✅ Safe"), (80, "🟡 Moderate"), (365, "🔴 Unhealthy"), (999, "🟣 Hazardous")],
    },
    "CO": {
        "name": "Carbon Monoxide (CO)",
        "desc": "Odorless, colorless gas. Called the 'silent killer'.",
        "who_limit": 4000, "unit": "μg/m³",
        "health": "Reduces blood oxygen capacity. High levels cause headache, unconsciousness, death.",
        "sources": "Incomplete combustion — vehicle exhaust, cooking fires, generators.",
        "risk_levels": [(4000, "✅ Safe"), (10000, "🟡 Moderate"), (30000, "🔴 Dangerous"), (999999, "🟣 Life-threatening")],
    },
    "OZONE": {
        "name": "Ground-level Ozone (O₃)",
        "desc": "Not the protective stratospheric ozone — this surface-level ozone is harmful.",
        "who_limit": 100, "unit": "μg/m³",
        "health": "Damages lung tissue, causes coughing and chest pain, worsens asthma and COPD.",
        "sources": "Formed by sunlight reacting with vehicle exhaust and industrial fumes (not directly emitted).",
        "risk_levels": [(100, "✅ Safe"), (160, "🟡 Moderate"), (240, "🔴 Unhealthy"), (999, "🟣 Hazardous")],
    },
    "NH3": {
        "name": "Ammonia (NH3)",
        "desc": "Pungent gas, major agricultural pollutant and precursor to secondary particles.",
        "who_limit": 400, "unit": "μg/m³",
        "health": "Irritates eyes, nose, throat. High levels cause severe lung damage.",
        "sources": "Chemical fertilizers, livestock waste, sewage treatment, industrial processes.",
        "risk_levels": [(400, "✅ Safe"), (700, "🟡 Moderate"), (999999, "🔴 Dangerous")],
    },
}

def risk_label(value, pollutant):
    """Return a human-readable risk label for a pollutant value."""
    info = POLLUTANT_INFO.get(pollutant, {})
    for threshold, label in info.get("risk_levels", [(9999, "Unknown")]):
        if value <= threshold:
            return label
    return "🟣 Hazardous"

def city_pollutant_avg(city, pollutant):
    sub = df[(df["city"] == city) & (df["pollutant_id"] == pollutant)]["pollutant_avg"].dropna()
    return round(sub.mean(), 1) if len(sub) else None

def state_pollutant_avg(state, pollutant):
    sub = df[(df["state"] == state) & (df["pollutant_id"] == pollutant)]["pollutant_avg"].dropna()
    return round(sub.mean(), 1) if len(sub) else None

def national_avg(pollutant):
    sub = df[df["pollutant_id"] == pollutant]["pollutant_avg"].dropna()
    return round(sub.mean(), 1) if len(sub) else None

def top_cities_for(pollutant, n=5, ascending=False, state_filter=None):
    sub = df[df["pollutant_id"] == pollutant]
    if state_filter:
        sub = sub[sub["state"] == state_filter]
    ranked = sub.groupby("city")["pollutant_avg"].mean().dropna()
    ranked = ranked.sort_values(ascending=ascending).head(n)
    return [(city, round(val, 1)) for city, val in ranked.items()]

def city_full_report(city):
    result = {}
    for pol in ALL_POLLUTANTS:
        val = city_pollutant_avg(city, pol)
        if val is not None:
            result[pol] = val
    return result

def state_full_report(state):
    result = {}
    for pol in ALL_POLLUTANTS:
        val = state_pollutant_avg(state, pol)
        if val is not None:
            result[pol] = val
    return result

def cities_in_state(state):
    return sorted(df[df["state"] == state]["city"].unique().tolist())

def top_states(n=5, ascending=False):
    sub = df[df["pollutant_id"] == "PM2.5"]
    ranked = sub.groupby("state")["pollutant_avg"].mean().dropna()
    ranked = ranked.sort_values(ascending=ascending).head(n)
    return [(s, round(v, 1)) for s, v in ranked.items()]

def safe_cities_for(pollutant="PM2.5", n=10):
    info = POLLUTANT_INFO.get(pollutant, {})
    limit = info.get("who_limit", 9999)
    sub = df[df["pollutant_id"] == pollutant]
    avgs = sub.groupby("city")["pollutant_avg"].mean().dropna()
    safe = avgs[avgs <= limit].sort_values().head(n)
    return [(c, round(v, 1)) for c, v in safe.items()]

# ════════════════════════════════════════════
# 6. RESPONSE BUILDER
# ════════════════════════════════════════════
def build_response(intent, cities, states, pollutants, n, original_text):
    """Build the actual answer based on detected intent and entities."""

    t = original_text.lower()

    # ── GREETING ──────────────────────────────
    if intent == "greeting":
        return (
            "👋 Hello! I'm AirBot, India's Air Quality Assistant.\n\n"
            "I have real-time data on 3,294 pollution readings across\n"
            "251 cities and 30 states — powered by ML algorithms.\n\n"
            "I can help you with:\n"
            "• Pollution levels for any city or state\n"
            "• Ranking cities by any pollutant (PM2.5, NO2, SO2...)\n"
            "• Comparing cities or states side by side\n"
            "• Health risk assessments\n"
            "• Full air quality reports\n"
            "• Explaining what each pollutant means\n\n"
            "Try asking:\n"
            "\"What is the PM2.5 level in Delhi?\"\n"
            "\"Compare Mumbai and Chennai air quality\"\n"
            "\"Which state has the worst air quality?\"\n"
            "\"Is Bengaluru safe to breathe?\""
        )

    # ── DATASET INFO ───────────────────────────
    if intent == "dataset_info":
        return (
            f"📊 Dataset Coverage\n\n"
            f"Total readings : {len(df):,}\n"
            f"Cities covered : {df['city'].nunique()}\n"
            f"States covered : {df['state'].nunique()}\n"
            f"Pollutants     : {', '.join(ALL_POLLUTANTS)}\n"
            f"Data date      : February 25, 2026\n"
            f"Source         : Central Pollution Control Board (CPCB), India"
        )

    # ── EXPLAIN POLLUTANT ─────────────────────
    if intent == "explain_pollutant":
        # Special: comparing two pollutants
        if "pm2.5" in t and "pm10" in t:
            return (
                "🔬 PM2.5 vs PM10 — What's the Difference?\n\n"
                "PM10 (Coarse particles, <10 μm)\n"
                "  • Dust, pollen, mold\n"
                "  • Stopped by nose and throat mostly\n"
                "  • WHO limit: 45 μg/m³\n\n"
                "PM2.5 (Fine particles, <2.5 μm)\n"
                "  • From vehicle exhaust, factory smoke\n"
                "  • Penetrate deep into lungs and blood\n"
                "  • WHO limit: 15 μg/m³ — more dangerous\n\n"
                "⚠️ PM2.5 is 3x more harmful than PM10.\n"
                "India's cities exceed PM2.5 limits by 5–20x in winter."
            )

        if pollutants:
            pol = pollutants[0]
            info = POLLUTANT_INFO[pol]
            nat = national_avg(pol)
            status = risk_label(nat, pol) if nat else "No data"
            return (
                f"🔬 {info['name']}\n\n"
                f"{info['desc']}\n\n"
                f"⚕️  Health effects : {info['health']}\n\n"
                f"🏭 Sources        : {info['sources']}\n\n"
                f"📏 WHO safe limit : {info['who_limit']} {info['unit']}\n"
                f"🇮🇳 India average  : {nat} {info['unit']} — {status}"
            )

    # ── AVERAGE: CITY + POLLUTANT ─────────────
    if intent == "average" and cities:
        results = []
        pols = pollutants if pollutants else ["PM2.5"]  # default to PM2.5
        for city in cities[:2]:  # max 2 cities
            for pol in pols[:3]:  # max 3 pollutants
                val = city_pollutant_avg(city, pol)
                if val:
                    info = POLLUTANT_INFO[pol]
                    status = risk_label(val, pol)
                    nat = national_avg(pol)
                    diff = round(val - nat, 1) if nat else None
                    vs_nat = f" (India avg: {nat}, {'▲ +' if diff and diff > 0 else '▼ '}{diff if diff else ''})" if diff is not None else ""
                    results.append(f"📍 {city} — {pol}\n   {val} {info['unit']} | {status}{vs_nat}")
                else:
                    results.append(f"📍 {city} — {pol}\n   No data available")
        if results:
            return "\n\n".join(results)

    # ── AVERAGE: STATE + POLLUTANT ────────────
    if intent == "average" and states and not cities:
        results = []
        pols = pollutants if pollutants else ["PM2.5"]
        for state in states[:2]:
            for pol in pols[:3]:
                val = state_pollutant_avg(state, pol)
                if val:
                    info = POLLUTANT_INFO[pol]
                    status = risk_label(val, pol)
                    results.append(f"🗺️  {state} — {pol}\n   {val} {info['unit']} | {status}")
                else:
                    results.append(f"🗺️  {state} — {pol}\n   No data available")
        if results:
            return "\n\n".join(results)

    # ── NATIONAL AVERAGE ──────────────────────
    if intent == "national_average":
        pols = pollutants if pollutants else ALL_POLLUTANTS
        lines = ["🇮🇳 India National Averages\n"]
        for pol in pols:
            val = national_avg(pol)
            info = POLLUTANT_INFO[pol]
            if val:
                status = risk_label(val, pol)
                lines.append(f"  {pol:6s}: {val} {info['unit']}  {status}")
        return "\n".join(lines)

    # ── TOP POLLUTED CITIES ───────────────────
    if intent == "top_polluted":
        pol = pollutants[0] if pollutants else "PM2.5"
        info = POLLUTANT_INFO[pol]
        state_filter = states[0] if states else None
        label = f"in {state_filter}" if state_filter else "in India"
        top = top_cities_for(pol, n=n, ascending=False, state_filter=state_filter)
        if not top:
            return f"No {pol} data found {label}."
        lines = [f"🏆 Top {len(top)} Most Polluted Cities {label} — {pol}\n"]
        for i, (city, val) in enumerate(top, 1):
            status = risk_label(val, pol)
            lines.append(f"  {i}. {city:20s} {val} {info['unit']}  {status}")
        lines.append(f"\n  WHO safe limit: {info['who_limit']} {info['unit']}")
        return "\n".join(lines)

    # ── LEAST POLLUTED CITIES ─────────────────
    if intent == "least_polluted":
        pol = pollutants[0] if pollutants else "PM2.5"
        info = POLLUTANT_INFO[pol]
        state_filter = states[0] if states else None
        label = f"in {state_filter}" if state_filter else "in India"
        bottom = top_cities_for(pol, n=n, ascending=True, state_filter=state_filter)
        if not bottom:
            return f"No {pol} data found {label}."
        lines = [f"🌿 Top {len(bottom)} Cleanest Cities {label} — {pol}\n"]
        for i, (city, val) in enumerate(bottom, 1):
            status = risk_label(val, pol)
            lines.append(f"  {i}. {city:20s} {val} {info['unit']}  {status}")
        return "\n".join(lines)

    # ── COMPARE CITIES ────────────────────────
    if intent == "compare" and len(cities) >= 2:
        c1, c2 = cities[0], cities[1]
        pols = pollutants if pollutants else ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE"]
        lines = [f"⚖️  {c1}  vs  {c2}\n"]
        for pol in pols:
            v1 = city_pollutant_avg(c1, pol)
            v2 = city_pollutant_avg(c2, pol)
            if v1 is None and v2 is None:
                continue
            info = POLLUTANT_INFO[pol]
            s1 = f"{v1} {info['unit']}" if v1 else "No data"
            s2 = f"{v2} {info['unit']}" if v2 else "No data"
            winner = ""
            if v1 and v2:
                winner = f"  ← cleaner" if v1 < v2 else ""
                w2 = f"  ← cleaner" if v2 < v1 else ""
                lines.append(f"  {pol:6s}: {s1:18s}{winner} | {s2:18s}{w2}")
            else:
                lines.append(f"  {pol:6s}: {s1:18s} | {s2}")
        return "\n".join(lines)

    # ── COMPARE STATES ────────────────────────
    if intent == "compare" and len(states) >= 2:
        s1, s2 = states[0], states[1]
        pols = pollutants if pollutants else ["PM2.5", "PM10", "NO2", "SO2"]
        lines = [f"⚖️  {s1}  vs  {s2}\n"]
        for pol in pols:
            v1 = state_pollutant_avg(s1, pol)
            v2 = state_pollutant_avg(s2, pol)
            if v1 is None and v2 is None:
                continue
            info = POLLUTANT_INFO[pol]
            s1v = f"{v1} {info['unit']}" if v1 else "No data"
            s2v = f"{v2} {info['unit']}" if v2 else "No data"
            winner1 = "  ← cleaner" if v1 and v2 and v1 < v2 else ""
            winner2 = "  ← cleaner" if v1 and v2 and v2 < v1 else ""
            lines.append(f"  {pol:6s}: {s1v:18s}{winner1} | {s2v:18s}{winner2}")
        return "\n".join(lines)

    # ── STATE RANKING ─────────────────────────
    if intent == "state_ranking":
        ascending = any(w in t for w in ["best", "cleanest", "least", "lowest", "safest"])
        label = "Cleanest" if ascending else "Most Polluted"
        ranked = top_states(n=n, ascending=ascending)
        lines = [f"🗺️  {label} States in India (by PM2.5)\n"]
        info = POLLUTANT_INFO["PM2.5"]
        for i, (state, val) in enumerate(ranked, 1):
            status = risk_label(val, "PM2.5")
            lines.append(f"  {i}. {state:25s} {val} {info['unit']}  {status}")
        return "\n".join(lines)

    # ── CITY FULL REPORT ──────────────────────
    if intent == "city_report" and cities:
        city = cities[0]
        report = city_full_report(city)
        if not report:
            return f"❌ No data found for {city}."
        station_count = df[df["city"] == city]["station"].nunique()
        lines = [f"📋 Full Air Quality Report — {city}\n",
                 f"  Monitoring stations: {station_count}\n"]
        for pol, val in report.items():
            info = POLLUTANT_INFO[pol]
            status = risk_label(val, pol)
            lines.append(f"  {pol:6s}: {val} {info['unit']:8s} {status}  (limit: {info['who_limit']})")
        return "\n".join(lines)

    # ── STATE FULL REPORT ─────────────────────
    if intent == "state_report" and states:
        state = states[0]
        report = state_full_report(state)
        if not report:
            return f"❌ No data found for {state}."
        city_count = df[df["state"] == state]["city"].nunique()
        lines = [f"📋 Full Air Quality Report — {state}\n",
                 f"  Cities monitored: {city_count}\n"]
        for pol, val in report.items():
            info = POLLUTANT_INFO[pol]
            status = risk_label(val, pol)
            lines.append(f"  {pol:6s}: {val} {info['unit']:8s} {status}  (limit: {info['who_limit']})")
        return "\n".join(lines)

    # ── HEALTH CHECK ──────────────────────────
    if intent == "health_check" and cities:
        city = cities[0]
        report = city_full_report(city)
        if not report:
            return f"❌ No data found for {city}."
        dangerous = [(p, v) for p, v in report.items()
                     if v > POLLUTANT_INFO[p]["who_limit"]]
        safe_pols  = [(p, v) for p, v in report.items()
                     if v <= POLLUTANT_INFO[p]["who_limit"]]

        lines = [f"🏥 Health Risk Assessment — {city}\n"]
        if not dangerous:
            lines.append("✅ All pollutants are within WHO safe limits!\n")
        else:
            lines.append(f"⚠️  {len(dangerous)} pollutant(s) exceed WHO limits:\n")
            for pol, val in dangerous:
                info = POLLUTANT_INFO[pol]
                times = round(val / info["who_limit"], 1)
                lines.append(f"  🔴 {pol}: {val} {info['unit']} ({times}x above safe limit)")
                lines.append(f"     Risk: {info['health'][:60]}...")
        if safe_pols:
            lines.append(f"\n✅ Within safe limits: {', '.join(p for p,v in safe_pols)}")
        return "\n".join(lines)

    # ── CITIES IN STATE ───────────────────────
    if intent == "cities_in_state" and states:
        state = states[0]
        city_list = cities_in_state(state)
        lines = [f"🏙️  Cities monitored in {state} ({len(city_list)} total)\n"]
        # Display in columns of 3
        for i in range(0, len(city_list), 3):
            row = city_list[i:i+3]
            lines.append("  " + "  |  ".join(f"{c:18s}" for c in row))
        return "\n".join(lines)

    # ── SAFE CITIES ───────────────────────────
    if intent == "safe_cities":
        pol = pollutants[0] if pollutants else "PM2.5"
        info = POLLUTANT_INFO[pol]
        safe = safe_cities_for(pol, n=10)
        if not safe:
            return f"⚠️ No cities currently meet the WHO safe limit for {pol} ({info['who_limit']} {info['unit']})."
        lines = [f"✅ Cities within WHO Safe Limit for {pol} (<{info['who_limit']} {info['unit']})\n"]
        for city, val in safe:
            lines.append(f"  • {city:20s} {val} {info['unit']}")
        return "\n".join(lines)

    # ── FALLBACK ──────────────────────────────
    hints = []
    if cities:
        hints.append(f'• "Full report for {cities[0]}"')
        hints.append(f'• "Is {cities[0]} safe to breathe?"')
    if states:
        hints.append(f'• "Top 5 polluted cities in {states[0]}"')
    if pollutants:
        hints.append(f'• "Which city has highest {pollutants[0]}?"')
    if not hints:
        hints = [
            '• "Average PM2.5 in Delhi"',
            '• "Compare Mumbai and Chennai"',
            '• "Worst state for air quality"',
            '• "Top 5 cities with highest NO2"',
            '• "Is Hyderabad safe to breathe?"',
        ]
    return (
        "🤔 I couldn't understand that clearly. Please try rephrasing.\n\n"
        "Example questions you can ask:\n" + "\n".join(hints)
    )

# ════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ════════════════════════════════════════════
def generate_response(user_input):
    """
    Main function — takes any user question, returns accurate answer.
    Pipeline:
      1. Spell correction
      2. Entity extraction (cities, states, pollutants, numbers)
      3. Keyword intent detection (fast, high accuracy)
      4. TF-IDF + Cosine Similarity fallback (ML)
      5. Response generation
    """
    # Step 1: Correct spelling
    corrected = correct_spelling(user_input)

    # Step 2: Extract entities
    cities     = find_all_cities(corrected)
    states     = find_all_states(corrected)
    pollutants = find_all_pollutants(corrected)
    n          = extract_number(corrected)

    # Step 3: Keyword-based intent (high accuracy, fast)
    intent = keyword_intent(corrected)

    # Step 4: TF-IDF fallback if keyword didn't match
    if not intent:
        intent, confidence = classify_intent(corrected)
        if intent == "unknown":
            # Last resort: infer from entities
            if cities and pollutants:
                intent = "average"
            elif cities:
                intent = "city_report"
            elif states and pollutants:
                intent = "average"
            elif states:
                intent = "state_report"

    # Step 5: Generate response
    return build_response(intent, cities, states, pollutants, n, corrected)
