import hashlib
import json
import os
import re
import threading
from typing import Any

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from utils import deep_get as _deep_get
from utils import first_fact_with as _first_fact_with


DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 8.0
COMPOSE_MODEL = (
    os.getenv("AZURE_OPENAI_COMPOSE_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("OPENAI_COMPOSE_MODEL")
    or "gpt-4.1"
)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_VOLATILE_TRIGGER_HASH_FIELDS = {
    "delivered_at",
    "received_at",
    "created_at",
    "updated_at",
    "timestamp",
    "ts",
}
_llm_client: AzureOpenAI | OpenAI | None = None
_llm_lock = threading.Lock()


def has_llm_credentials() -> bool:
    return bool(
        (os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
        or os.getenv("OPENAI_API_KEY")
    )


def get_llm_client() -> AzureOpenAI | OpenAI:
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    with _llm_lock:
        if _llm_client is not None:
            return _llm_client
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        try:
            timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS))
        except ValueError:
            timeout = DEFAULT_OPENAI_TIMEOUT_SECONDS
        if azure_api_key:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT is required when AZURE_OPENAI_API_KEY is set")
            _llm_client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
                azure_endpoint=azure_endpoint,
                timeout=timeout,
            )
        else:
            _llm_client = OpenAI(timeout=timeout)
    return _llm_client


TRIGGER_ROUTE_MAP = {
    "research_digest": "research",
    "category_research": "research",
    "category_research_digest_release": "research",
    "regulation_change": "research",
    "cde_opportunity": "research",
    "supply_alert": "research",
    "recall_due": "recall",
    "customer_lapsed_soft": "recall",
    "customer_lapsed_hard": "recall",
    "appointment_tomorrow": "recall",
    "chronic_refill_due": "recall",
    "trial_followup": "recall",
    "wedding_package_followup": "recall",
    "perf_dip": "perf_dip",
    "seasonal_perf_dip": "perf_dip",
    "perf_spike": "perf_spike",
    "milestone_reached": "milestone",
    "festival_upcoming": "festival",
    "ipl_match_today": "festival",
    "dormant_with_vera": "reactivation",
    "winback_eligible": "reactivation",
    "renewal_due": "reactivation",
    "review_theme_emerged": "review_insight",
    "competitor_opened": "competitive",
    "curious_ask_due": "curious_ask",
    "scheduled_recurring": "curious_ask",
    "active_planning_intent": "curious_ask",
    "stale_posts": "content_nudge",
    "ctr_below_peer": "content_nudge",
    "category_seasonal": "content_nudge",
    "gbp_unverified": "content_nudge",
}


ROUTE_KEYWORD_FALLBACKS = (
    ("perf_spike", ("spike", "surge", "growth")),
    ("perf_dip", ("dip", "drop", "below_peer", "decline")),
    ("recall", ("recall", "lapsed", "refill", "appointment", "trial", "followup", "follow_up", "wedding")),
    ("research", ("research", "digest", "regulation", "compliance", "cde", "webinar", "supply", "alert")),
    ("festival", ("festival", "diwali", "holi", "ipl", "match", "event")),
    ("milestone", ("milestone",)),
    ("review_insight", ("review", "theme")),
    ("competitive", ("competitor", "competitive")),
    ("curious_ask", ("curious", "ask", "planning", "intent")),
    ("reactivation", ("dormant", "winback", "renewal", "quiet")),
    ("content_nudge", ("content", "post", "gbp", "unverified", "seasonal", "ctr")),
)


ROUTE_INSTRUCTIONS = {
    "research": (
        "Lead with the finding. Include: trial size + % stat + source citation. "
        "Connect to THIS merchant's specific customer cohort. Offer to draft "
        "shareable customer education content. Match the category voice; for "
        "clinical categories use peer/clinical tone. No hype."
    ),
    "recall": (
        "This is a CUSTOMER-facing message (send_as=merchant_on_behalf). Include: "
        "customer first name, exact months since last visit, real available slots "
        "from merchant offers/schedule, service price. Match customer's language "
        "preference (hi-en mix if indicated). CTA: Reply 1 for slot A, 2 for slot B."
    ),
    "perf_dip": (
        "Lead with specific metric drop (exact %). Frame as loss aversion: 'you "
        "are missing X'. Offer ONE concrete fix tied to their existing offers. "
        "Binary YES CTA. Under 200 chars."
    ),
    "perf_spike": (
        "Celebrate the specific metric spike. Propose capitalizing NOW - double "
        "down on what's working. Use social proof: '3 merchants in your locality "
        "saw similar spikes after X'. Offer to draft follow-up campaign."
    ),
    "milestone": (
        "Celebrate the named milestone with the exact number/date. Convert the "
        "moment into one next action: a Google post, WhatsApp status, or customer "
        "thank-you. Keep it specific and avoid generic congratulations."
    ),
    "festival": (
        "Name the festival + days remaining. Reference their existing offer or "
        "propose one from category offer_catalog. Binary YES/STOP CTA. Urgency "
        "without pressure."
    ),
    "reactivation": (
        "Merchant has been quiet. Do NOT ask why. Lead with something immediately "
        "useful - a digest item, a stat they haven't seen, a profile gap. "
        "Curiosity hook. Under 160 chars."
    ),
    "review_insight": (
        "Name the specific review theme. Quote the pattern (without naming a "
        "customer). Offer a templated response they can use. Asking-the-merchant "
        "lever: 'Is this a real pain point?'"
    ),
    "competitive": (
        "Name the competitor signal if present. Reframe as opportunity. Surface "
        "THIS merchant's differentiator (their rating vs peer avg, their unique "
        "offer). Curiosity CTA: 'Want to see how you compare?'"
    ),
    "curious_ask": (
        "Ask ONE specific open question about their business this week. Use the "
        "'asking the merchant' lever. Offer to turn the answer into a deliverable "
        "(Google post, WhatsApp reply template, campaign brief). Keep it "
        "conversational. Under 180 chars."
    ),
    "content_nudge": (
        "Reference the specific signal (e.g. 'last post 22 days ago'). Offer to "
        "draft the content. Effort externalization: 'I'll write it - just say go'."
    ),
    "generic": (
        "Use the most relevant real metric or offer. Explain why the message is "
        "being sent now. Ask for one simple reply action."
    ),
}


FEW_SHOT_LIBRARY: dict[str, list[dict[str, str]]] = {
    "research": [
        {
            "input": "dentist, research_digest trigger, JIDA Oct paper, high-risk adult patients",
            "output": (
                "Dr. Meera, JIDA's Oct issue landed. One item for your high-risk "
                "adult patients - 2,100-patient trial: 3-month fluoride recall "
                "cuts caries 38% better than 6-month. Want me to pull it + draft "
                "a patient-ed WhatsApp? - JIDA Oct 2026 p.14"
            ),
            "score": "50/50",
        },
        {
            "input": "pharmacy, cough-season digest, IMA advisory",
            "output": (
                "Anil, IMA's Nov note flags a 27% rise in dry-cough visits in "
                "Delhi clinics. Useful for your pharmacy customers: want me to "
                "draft a compliance-safe Google post on when to consult a doctor?"
            ),
            "score": "47/50",
        },
    ],
    "recall": [
        {
            "input": "dentist, patient Riya, 8 months since cleaning, slots Fri 5pm/Sat 11am, cleaning Rs299",
            "output": (
                "Hi Riya, this is Dr. Kapoor's clinic. It's been 8 months since "
                "your cleaning. We have Fri 5pm or Sat 11am; cleaning is ₹299. "
                "Reply 1 for Fri, 2 for Sat."
            ),
            "score": "48/50",
        },
        {
            "input": "salon, customer Neha, 5 months since facial, slots Tue 4pm/Wed 1pm, glow facial Rs799",
            "output": (
                "Hi Neha, Studio11 here. Your last facial was 5 months ago; glow "
                "facial is ₹799 this week. Tue 4pm or Wed 1pm available. Reply "
                "1 for Tue, 2 for Wed."
            ),
            "score": "46/50",
        },
    ],
    "perf_dip": [
        {
            "input": "restaurant, calls dropped 40% WoW",
            "output": (
                "Suresh, ek problem: aapke calls 40% gire hain is hafte. Main "
                "dekh rahi hoon - title outdated lag raha hai. Main abhi fix "
                "karti hoon? 5 min kaam hai. Reply YES."
            ),
            "score": "46/50",
        },
        {
            "input": "gym, direction requests down 31%, trial class offer active",
            "output": (
                "Rohit, direction requests 31% down this week - you may be losing "
                "trial walk-ins. I can refresh the 'Free Trial Class' post now. "
                "Reply YES."
            ),
            "score": "45/50",
        },
    ],
    "perf_spike": [
        {
            "input": "restaurant, profile views up 24%, biryani offer active",
            "output": (
                "Aamir, profile views are up 24% this week. 3 nearby restaurants "
                "converted similar spikes with a lunch post. Want me to draft one "
                "for your Biryani Combo @ ₹199?"
            ),
            "score": "46/50",
        },
        {
            "input": "salon, calls up 18%, hair spa offer",
            "output": (
                "Lakshmi, calls are up 18% after your hair spa offer. Let's use "
                "the momentum today - I'll draft a follow-up WhatsApp for Hair "
                "Spa @ ₹999. Say GO?"
            ),
            "score": "45/50",
        },
    ],
    "milestone": [
        {
            "input": "dentist, 500 reviews milestone, rating 4.8",
            "output": (
                "Dr. Jain, you crossed 500 Google reviews at 4.8. Strong trust "
                "signal for new patients - want me to draft a thank-you post for "
                "your profile today?"
            ),
            "score": "45/50",
        },
        {
            "input": "restaurant, 1,000 direction requests milestone",
            "output": (
                "Nisha, your cafe just crossed 1,000 direction requests this "
                "month. Nice local demand signal. Want a 3-line Google post to "
                "turn it into weekend visits?"
            ),
            "score": "44/50",
        },
    ],
    "festival": [
        {
            "input": "salon, Diwali in 6 days, party makeup offer",
            "output": (
                "Lakshmi, Diwali is 6 days away and your Party Makeup @ ₹1499 is "
                "active. I can draft a last-slots WhatsApp campaign now. Reply "
                "YES or STOP."
            ),
            "score": "46/50",
        },
        {
            "input": "restaurant, Holi in 4 days, family thali offer",
            "output": (
                "Manoj, Holi is 4 days away. Your Family Thali @ ₹599 fits group "
                "orders perfectly - want me to draft a festive Google post? YES/STOP"
            ),
            "score": "44/50",
        },
    ],
    "reactivation": [
        {
            "input": "dentist, dormant 21 days, digest item on fluoride recall",
            "output": (
                "Dr. Rao, one useful JIDA note: 3-month fluoride recall beat "
                "6-month by 38%. I can turn it into a patient WhatsApp in 5 min."
            ),
            "score": "45/50",
        },
        {
            "input": "restaurant, dormant 14 days, posts stale",
            "output": (
                "Priya, your last Google post is 22 days old while searches are "
                "up this week. I can write a fresh lunch post - just say GO."
            ),
            "score": "44/50",
        },
    ],
    "review_insight": [
        {
            "input": "dentist, reviews mention waiting time 9 times",
            "output": (
                "Dr. Shah, 9 recent reviews mention 'waiting time'. Is this a real "
                "pain point? I can draft a calm reply template that protects trust."
            ),
            "score": "46/50",
        },
        {
            "input": "salon, reviews praise staff but mention pricing confusion",
            "output": (
                "Lakshmi, reviews praise your staff but 6 mention 'price confusion'. "
                "Is that happening often? I can draft a clear reply + price FAQ."
            ),
            "score": "45/50",
        },
    ],
    "competitive": [
        {
            "input": "restaurant, competitor opened nearby, merchant rating 4.6 vs peer 4.2",
            "output": (
                "Suresh, a new cafe opened 400m away. Opportunity: your rating is "
                "4.6 vs local peer avg 4.2. Want to see how you compare?"
            ),
            "score": "45/50",
        },
        {
            "input": "gym, competitor added trial offer, merchant has 7-day pass",
            "output": (
                "Rohit, FitZone just added a trial offer nearby. Your edge: 7-day "
                "Pass @ ₹499 already beats a one-day trial. Want to see the gap?"
            ),
            "score": "44/50",
        },
    ],
    "curious_ask": [
        {
            "input": "salon, weekly cadence",
            "output": (
                "Hi Lakshmi! Quick check - what service has been most asked-for "
                "this week at Studio11? I'll turn the answer into a Google post + "
                "a 4-line WhatsApp reply for pricing questions. Takes 5 min."
            ),
            "score": "44/50",
        },
        {
            "input": "restaurant, weekly ask, dinner demand",
            "output": (
                "Suresh, quick ask: which item got the most dinner enquiries this "
                "week? Tell me one dish and I'll draft a Google post + WhatsApp "
                "reply to push it tonight."
            ),
            "score": "44/50",
        },
    ],
    "content_nudge": [
        {
            "input": "dentist, last post 28 days ago, cleaning offer active",
            "output": (
                "Dr. Meera, your last Google post was 28 days ago. I can write a "
                "fresh post around Dental Cleaning @ ₹299 - just say GO."
            ),
            "score": "45/50",
        },
        {
            "input": "restaurant, CTR 2.1 vs peer 3.0, stale menu photo",
            "output": (
                "Arjun, CTR is 2.1% vs peer 3.0%. Your menu photo looks stale; "
                "I'll write a new weekend post for Paneer Combo @ ₹179. Say GO."
            ),
            "score": "44/50",
        },
    ],
    "generic": [
        {
            "input": "pharmacy, views up 12%, OTC safety post due",
            "output": (
                "Anil, views are up 12% this week. Good time for a compliance-safe "
                "OTC safety post. Want me to draft one for your Google profile?"
            ),
            "score": "42/50",
        },
        {
            "input": "gym, new year enquiries rising, trial offer active",
            "output": (
                "Rohit, New Year fitness searches are rising and your Trial Week "
                "@ ₹499 is active. Want me to draft a 4-line WhatsApp campaign?"
            ),
            "score": "42/50",
        },
    ],
}


def route_for_trigger(trigger: dict[str, Any]) -> str:
    kind = (
        trigger.get("kind")
        or trigger.get("type")
        or trigger.get("event")
        or trigger.get("trigger_kind")
        or ""
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", str(kind).strip().lower()).strip("_")
    if normalized in TRIGGER_ROUTE_MAP:
        return TRIGGER_ROUTE_MAP[normalized]
    for route, keywords in ROUTE_KEYWORD_FALLBACKS:
        if any(keyword in normalized for keyword in keywords):
            return route
    return "generic"


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            items.extend(_flatten(value, f"{prefix}[{idx}]"))
    else:
        items.append((prefix, obj))
    return items


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fmt_percent(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if "%" in stripped:
            return stripped
        match = re.search(r"-?\d+(?:\.\d+)?", stripped)
        if not match:
            return None
        value = float(match.group(0))
    if isinstance(value, (int, float)):
        number = float(value)
        if abs(number) <= 1:
            number *= 100
        elif abs(number) > 200:
            number /= 100
        return f"{number:.1f}%".replace(".0%", "%")
    return None


def _fmt_money(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "₹" in text or text.lower().startswith("rs"):
        return text.replace("Rs.", "₹").replace("Rs ", "₹")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        amount = match.group(0)
        if amount.endswith(".0"):
            amount = amount[:-2]
        return f"₹{amount}"
    return None


def _merchant_name(merchant: dict[str, Any] | None) -> str:
    value = _deep_get(
        merchant,
        "owner_name",
        "merchant_name",
        "business_name",
        "name",
        "identity.owner_name",
        "identity.business_name",
    )
    return str(value or "there").strip()


def _customer_name(customer: dict[str, Any] | None) -> str:
    value = _deep_get(customer, "first_name", "name", "profile.first_name", "profile.name")
    if not value:
        return "there"
    return str(value).split()[0]


def _first_offer(
    merchant: dict[str, Any] | None,
    category: dict[str, Any] | None,
    trigger: dict[str, Any] | None = None,
) -> str | None:
    offer_sources = [
        _deep_get(merchant, "active_offers", "offers", "offers.active", "campaigns.active_offers"),
        _deep_get(category, "offer_catalog", "offers", "recommended_offers"),
    ]
    trigger_text = json.dumps(trigger or {}, ensure_ascii=False).lower()
    trigger_keywords = {
        token
        for token in re.split(r"[^a-z0-9₹]+", trigger_text)
        if len(token) >= 3 and token not in {"true", "false", "null", "kind", "payload"}
    }
    candidates: list[tuple[int, str]] = []
    for offers in offer_sources:
        for offer in _as_list(offers):
            if isinstance(offer, dict):
                status = str(offer.get("status") or offer.get("state") or "").lower()
                if status and any(s in status for s in ("inactive", "expired", "paused")):
                    continue
                name = offer.get("name") or offer.get("title") or offer.get("service")
                price = _fmt_money(
                    offer.get("price")
                    or offer.get("amount")
                    or offer.get("discounted_price")
                    or offer.get("value")
                )
                label = ""
                if name and price:
                    suffix = " (active)" if not status or "active" in status else ""
                    label = f"{name} @ {price}{suffix}"
                elif name:
                    suffix = " (active)" if not status or "active" in status else ""
                    label = f"{name}{suffix}"
                if label:
                    label_l = label.lower()
                    score = 1 if "active" in label_l or not status else 0
                    score += sum(4 for token in trigger_keywords if token in label_l)
                    candidates.append((score, label))
            elif isinstance(offer, str) and offer.strip():
                label = offer.strip()
                label_l = label.lower()
                score = sum(4 for token in trigger_keywords if token in label_l)
                candidates.append((score, label))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _first_slot(*contexts: dict[str, Any] | None) -> tuple[str | None, str | None]:
    slots: list[Any] = []
    for ctx in contexts:
        slots.extend(
            _as_list(
                _deep_get(
                    ctx,
                    "available_slots",
                    "schedule.available_slots",
                    "availability.slots",
                    "slots",
                )
            )
        )
    labels: list[str] = []
    for slot in slots:
        if isinstance(slot, dict):
            label = (
                slot.get("label")
                or " ".join(
                    str(part)
                    for part in [slot.get("day"), slot.get("date"), slot.get("time")]
                    if part
                )
            )
        else:
            label = str(slot) if slot else ""
        if label:
            labels.append(label.strip())
    first = labels[0] if labels else None
    second = labels[1] if len(labels) > 1 else None
    return first, second


def _extract_digest_item(category: dict[str, Any] | None, trigger: dict[str, Any] | None) -> str | None:
    digest_sources = [
        _deep_get(
            trigger,
            "payload.top_item",
            "top_item",
            "digest_item",
            "top_digest_item",
            "research_item",
        ),
        _deep_get(category, "digest", "digest_items", "research_digest.items", "research.items"),
    ]
    for source in digest_sources:
        for item in _as_list(source):
            if isinstance(item, dict):
                title = item.get("title") or item.get("finding") or item.get("summary")
                source_name = item.get("source") or item.get("citation") or item.get("journal")
                stat = item.get("stat") or item.get("metric") or item.get("result")
                sample = item.get("n") or item.get("sample_size") or item.get("trial_size")
                page = item.get("page") or item.get("page_ref")
                parts = []
                if source_name:
                    parts.append(str(source_name))
                    if stat:
                        parts.append(str(stat))
                    if title:
                        parts.append(str(title))
                else:
                    if title:
                        parts.append(str(title))
                    if stat:
                        parts.append(str(stat))
                if sample:
                    parts.append(f"n={sample}")
                if page:
                    parts.append(f"p.{page}")
                if parts:
                    return ": ".join(parts[:2]) + (" (" + ", ".join(parts[2:]) + ")" if len(parts) > 2 else "")
            elif isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _generic_number_facts(
    merchant: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    category: dict[str, Any] | None,
    customer: dict[str, Any] | None,
) -> list[str]:
    contexts = [
        ("merchant", merchant),
        ("trigger", trigger),
        ("category", category),
        ("customer", customer),
    ]
    facts: list[str] = []
    for label, ctx in contexts:
        for path, value in _flatten(ctx or {}):
            if len(facts) >= 8:
                return facts
            path_l = path.lower()
            if isinstance(value, (int, float)) and any(
                token in path_l
                for token in (
                    "drop",
                    "growth",
                    "views",
                    "calls",
                    "rating",
                    "ctr",
                    "review",
                    "days",
                    "months",
                    "lapsed",
                    "price",
                    "distance",
                    "requests",
                )
            ):
                facts.append(f"{label} {path}: {value}")
            elif isinstance(value, str) and re.search(r"\d", value) and any(
                token in path_l
                for token in (
                    "source",
                    "citation",
                    "digest",
                    "metric",
                    "signal",
                    "offer",
                    "review",
                    "trend",
                    "festival",
                    "slot",
                    "price",
                )
            ):
                facts.append(f"{label} {path}: {value}")
    return facts


def extract_key_facts(
    category: dict[str, Any] | None,
    merchant: dict[str, Any] | None,
    trigger: dict[str, Any] | None,
    customer: dict[str, Any] | None = None,
) -> list[str]:
    facts: list[str] = []

    merchant_ctr = _deep_get(
        merchant,
        "metrics.ctr",
        "performance.ctr",
        "google_profile.ctr",
        "ctr",
    )
    peer_ctr = _deep_get(
        category,
        "peer_stats.avg_ctr",
        "peer_stats.ctr",
        "benchmarks.ctr",
        "peer.ctr",
        "metrics.peer_ctr",
    ) or _deep_get(trigger, "peer_ctr", "metrics.peer_ctr")
    if merchant_ctr is not None and peer_ctr is not None:
        m_ctr = _fmt_percent(merchant_ctr)
        p_ctr = _fmt_percent(peer_ctr)
        if m_ctr and p_ctr:
            facts.append(f"CTR gap: merchant {m_ctr} vs peer {p_ctr}")

    avg_rating = _deep_get(category, "peer_stats.avg_rating")
    avg_reviews = _deep_get(category, "peer_stats.avg_reviews")
    merchant_rating = _deep_get(
        merchant,
        "performance.google_rating",
        "performance.rating",
        "metrics.rating",
        "google_profile.rating",
        "identity.rating",
        "rating",
    )
    merchant_reviews = _deep_get(
        merchant,
        "performance.review_count",
        "metrics.review_count",
        "google_profile.review_count",
        "reviews_count",
    )
    if merchant_rating and avg_rating:
        facts.append(f"Rating: merchant {merchant_rating} vs peer avg {avg_rating}")
    if merchant_reviews and avg_reviews:
        facts.append(f"Reviews: merchant {merchant_reviews} vs peer avg {avg_reviews}")

    lapsed = _deep_get(
        trigger,
        "lapsed_patients",
        "signals.lapsed_patients",
        "lapsed_customers",
        "signals.lapsed_customers",
    ) or _deep_get(merchant, "signals.lapsed_patients", "signals.lapsed_customers")
    days = _deep_get(trigger, "lapsed_days", "days_since_last_visit", "threshold_days") or 180
    if lapsed:
        noun = "patients" if "dent" in json.dumps(category or {}).lower() else "customers"
        facts.append(f"Lapsed {noun}: {lapsed} lapsed >{days} days")

    lapsed_agg = _deep_get(merchant, "customer_aggregate.lapsed_180d_plus")
    if lapsed_agg:
        facts.append(f"Lapsed customers: {lapsed_agg} lapsed >180 days")

    high_risk = _deep_get(merchant, "customer_aggregate.high_risk_adult_count")
    if high_risk:
        facts.append(f"High-risk adult patients: {high_risk}")

    retention = _deep_get(merchant, "customer_aggregate.retention_6mo_pct")
    if retention:
        pct = _fmt_percent(retention)
        if pct:
            facts.append(f"6-month retention: {pct}")

    digest = _extract_digest_item(category, trigger)
    if digest:
        facts.append(f"Top digest item: {digest}")

    offer = _first_offer(merchant, category, trigger)
    if offer:
        facts.append(f"Active offers: {offer}")

    view_trend = _deep_get(
        trigger,
        "view_trend",
        "metrics.view_trend",
        "views_change",
        "views_wow",
    ) or _deep_get(merchant, "metrics.views_wow", "performance.views_wow")
    if view_trend:
        facts.append(f"View trends: {view_trend}")

    seasonal = _deep_get(
        trigger,
        "seasonal_signal",
        "signals.seasonal",
        "seasonal",
    ) or _deep_get(category, "seasonal_signal", "signals.seasonal")
    if seasonal:
        facts.append(f"Seasonal signal: {seasonal}")

    drop = _deep_get(trigger, "metric_drop", "drop", "metrics.drop_percent", "calls_drop")
    metric_name = _deep_get(trigger, "metric_name", "metric", "signal_name") or "metric"
    if drop:
        pct = _fmt_percent(drop) or str(drop)
        facts.append(f"Performance dip: {metric_name} down {pct}")

    spike = _deep_get(trigger, "metric_spike", "spike", "metrics.spike_percent", "views_spike")
    if spike:
        pct = _fmt_percent(spike) or str(spike)
        facts.append(f"Performance spike: {metric_name} up {pct}")

    festival = _deep_get(trigger, "festival", "festival_name", "event_name")
    days_remaining = _deep_get(trigger, "days_remaining", "days_to_festival")
    if festival:
        if days_remaining is not None:
            facts.append(f"Festival: {festival} in {days_remaining} days")
        else:
            facts.append(f"Festival: {festival}")

    review_theme = _deep_get(
        trigger,
        "review_theme",
        "theme",
        "signals.review_theme",
        "review_insight.theme",
    )
    review_count = _deep_get(trigger, "review_count", "signals.review_count", "theme_count")
    if review_theme:
        if review_count:
            facts.append(f"Review theme: {review_theme} mentioned {review_count} times")
        else:
            facts.append(f"Review theme: {review_theme}")

    competitor = _deep_get(
        trigger,
        "competitor_name",
        "competitor.name",
        "signals.competitor_name",
    )
    if competitor:
        distance = _deep_get(trigger, "competitor.distance", "distance", "distance_m")
        if distance:
            facts.append(f"Competitor signal: {competitor} opened {distance} away")
        else:
            facts.append(f"Competitor signal: {competitor}")

    last_post_days = _deep_get(
        trigger,
        "last_post_days",
        "days_since_last_post",
        "signals.days_since_last_post",
    ) or _deep_get(merchant, "signals.days_since_last_post", "google_profile.days_since_last_post")
    if last_post_days:
        facts.append(f"Content signal: last post {last_post_days} days ago")

    months_since = _deep_get(
        trigger,
        "months_since_last_visit",
        "months_since_last_order",
        "signals.months_since_last_visit",
    ) or _deep_get(customer, "months_since_last_visit", "months_since_last_order")
    if months_since:
        facts.append(f"Customer timing: {months_since} months since last visit")

    slot_a, slot_b = _first_slot(trigger, merchant)
    if slot_a and slot_b:
        facts.append(f"Available slots: {slot_a}, {slot_b}")
    elif slot_a:
        facts.append(f"Available slot: {slot_a}")

    for generic in _generic_number_facts(merchant, trigger, category, customer):
        if generic not in facts:
            facts.append(generic)
        if len(facts) >= 12:
            break

    return facts[:12]


def _json_compact(obj: Any, limit: int = 6000) -> str:
    text = json.dumps(obj or {}, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


def _voice_summary(category: dict[str, Any] | None) -> str:
    voice = _deep_get(category, "voice") or {}
    if not isinstance(voice, dict):
        return str(voice)
    return _json_compact(voice, 1200)


def _language_values(*contexts: dict[str, Any] | None) -> list[str]:
    languages: list[str] = []
    for ctx in contexts:
        value = _deep_get(
            ctx,
            "language",
            "languages",
            "preferred_language",
            "profile.language",
            "voice.languages",
        )
        for item in _as_list(value):
            if item:
                languages.append(str(item).lower())
    return languages


def _needs_hinglish(*contexts: dict[str, Any] | None) -> bool:
    langs = " ".join(_language_values(*contexts))
    return "hi" in langs or "hinglish" in langs


def build_prompts(
    route: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None,
    key_facts: list[str],
) -> tuple[str, str]:
    few_shots = FEW_SHOT_LIBRARY.get(route, FEW_SHOT_LIBRARY["generic"])
    key_fact_block = "\n".join(f"- {fact}" for fact in key_facts) or "- No numeric facts found"
    system_prompt = f"""
You are Vera, magicpin's AI assistant for Indian merchants on WhatsApp.

Route: {route}
Route-specific instruction: {ROUTE_INSTRUCTIONS.get(route, ROUTE_INSTRUCTIONS["generic"])}

Hard constraints:
- Return JSON only with keys: body, cta, send_as, suppression_key, rationale.
- body length must be <= 320 characters; aim for 150-200.
- No URLs.
- Use exactly one CTA.
- send_as must be "vera" or "merchant_on_behalf".
- Use at least one KEY FACT verbatim or nearly verbatim in the body.
- Do not fabricate numbers, dates, research citations, competitor names, offers, slots, prices, ratings, or peer stats.
- Do not use taboo vocabulary from the category voice.
- If language includes "hi", prefer one natural Hindi/Hinglish phrase.

Category voice:
{_voice_summary(category)}
""".strip()

    user_prompt = f"""
KEY FACTS:
{key_fact_block}

FEW-SHOT EXAMPLES FOR THIS ROUTE:
{_json_compact(few_shots, 2500)}

CATEGORY CONTEXT:
{_json_compact(category)}

MERCHANT CONTEXT:
{_json_compact(merchant)}

TRIGGER CONTEXT:
{_json_compact(trigger)}

CUSTOMER CONTEXT:
{_json_compact(customer)}

Compose the WhatsApp message now. Keep it specific to the merchant and trigger.
""".strip()
    return system_prompt, user_prompt


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def _chat_json(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> dict[str, Any]:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def _taboo_words(category: dict[str, Any] | None) -> list[str]:
    words: list[str] = []
    for value in (
        _deep_get(category, "voice.vocab_taboo"),
        _deep_get(category, "vocab_taboo"),
        _deep_get(category, "voice.taboo_words"),
    ):
        for word in _as_list(value):
            if word:
                words.append(str(word).strip())
    return [word for word in words if word]


def _validation_errors(result: dict[str, Any], category: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    body = str(result.get("body") or "")
    if len(body) > 320:
        errors.append("body_over_320_chars")
    if URL_RE.search(body):
        errors.append("url_in_body")
    body_l = body.lower()
    for word in _taboo_words(category):
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        if re.search(pattern, body_l):
            errors.append(f"taboo_word:{word}")
    if result.get("send_as") not in {"vera", "merchant_on_behalf"}:
        errors.append("invalid_send_as")
    if not result.get("cta"):
        errors.append("missing_cta")
    return errors


def _repair_result(result: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    system_prompt = result.get("_system_prompt")
    user_prompt = result.get("_user_prompt")
    if not system_prompt or not user_prompt:
        raise ValueError("missing repair prompt")
    repair_prompt = f"""
Your previous JSON violated: {", ".join(errors)}.
Rewrite only the JSON result. Preserve the same route and facts.
Absolute requirements:
- body <= 320 characters
- no URLs
- no taboo words
- send_as is vera or merchant_on_behalf
- one CTA only

Previous result:
{_json_compact({k: v for k, v in result.items() if not k.startswith("_")})}
""".strip()
    fixed = _chat_json(COMPOSE_MODEL, system_prompt, user_prompt + "\n\n" + repair_prompt)
    fixed["_system_prompt"] = system_prompt
    fixed["_user_prompt"] = user_prompt
    fixed["_route"] = result.get("_route")
    fixed["_key_facts"] = result.get("_key_facts")
    return fixed


def _final_cleanup(
    result: dict[str, Any], category: dict[str, Any], trigger: dict[str, Any]
) -> dict[str, Any]:
    body = str(result.get("body") or "").strip()
    body = URL_RE.sub("", body).strip()
    for word in _taboo_words(category):
        body = re.sub(r"\b" + re.escape(word) + r"\b", "", body, flags=re.IGNORECASE).strip()
    if len(body) > 320:
        body = body[:317].rstrip() + "..."

    send_as = result.get("send_as")
    if send_as not in {"vera", "merchant_on_behalf"}:
        send_as = "merchant_on_behalf" if route_for_trigger(trigger) == "recall" else "vera"

    cta = str(result.get("cta") or "").strip()
    if not cta:
        cta = "Reply YES" if send_as == "vera" else "Reply 1 or 2"

    clean = {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": str(result.get("suppression_key") or ""),
        "rationale": str(result.get("rationale") or "Composed from route context and key facts."),
    }
    if result.get("_route"):
        clean["route"] = result["_route"]
    if result.get("_key_facts"):
        clean["key_facts"] = result["_key_facts"]
    return clean


def _hard_safe_result(result: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    send_as = "merchant_on_behalf" if route_for_trigger(trigger) == "recall" else "vera"
    cta = "Reply 1 or 2" if send_as == "merchant_on_behalf" else "Reply YES"
    facts = result.get("_key_facts") or []
    fact = str(facts[0]) if facts else "your latest signal"
    merchant_name = str(result.get("_merchant_name") or "").strip()
    prefix = f"{merchant_name}, " if merchant_name else ""
    return {
        "body": _body_with_limit(f"{prefix}one signal is ready: {fact}. Want me to draft the message?"),
        "cta": cta,
        "send_as": send_as,
        "suppression_key": str(result.get("suppression_key") or ""),
        "rationale": "Returned hard-safe fallback after validation cleanup still failed.",
    }


def validate_and_fix(
    result: dict[str, Any],
    category: dict[str, Any],
    trigger: dict[str, Any],
    fallback_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best = result
    for _ in range(2):
        errors = _validation_errors(best, category)
        if not errors:
            return _final_cleanup(best, category, trigger)
        try:
            best = _repair_result(best, errors)
        except Exception:
            break
    cleaned = _final_cleanup(best, category, trigger)
    if not _validation_errors(cleaned, category):
        return cleaned
    if fallback_result:
        fallback_cleaned = _final_cleanup(fallback_result, category, trigger)
        if not _validation_errors(fallback_cleaned, category):
            return fallback_cleaned
    hard_cleaned = _final_cleanup(_hard_safe_result(best, trigger), category, trigger)
    if _validation_errors(hard_cleaned, category):
        hard_cleaned["body"] = "Reply YES to continue."
        hard_cleaned["cta"] = "Reply YES"
    return hard_cleaned


def _body_with_limit(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _fallback_message(
    route: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None,
    facts: list[str],
) -> dict[str, Any]:
    name = _merchant_name(merchant)
    customer_first = _customer_name(customer)
    offer = _first_offer(merchant, category, trigger) or "your active offer"
    fact = _first_fact_with(["performance", "ctr", "view", "festival", "review", "digest", "content"], facts)
    fact = fact or (facts[0] if facts else "your latest Google profile signal")
    hinglish = _needs_hinglish(category, merchant, customer)

    if route == "research":
        body = f"{name}, {fact}. Want me to pull it and draft a patient-ed WhatsApp you can share today?"
        cta = "Reply YES"
    elif route == "recall":
        months = _first_fact_with(["customer timing"], facts) or "it has been a while"
        slot_a, slot_b = _first_slot(trigger, merchant)
        slot_a = slot_a or "today 5pm"
        slot_b = slot_b or "tomorrow 11am"
        body = (
            f"Hi {customer_first}, {name} here. {months}. {offer}. "
            f"{slot_a} or {slot_b} available. Reply 1 for first, 2 for second."
        )
        cta = "Reply 1 or 2"
    elif route == "perf_dip":
        body = f"{name}, {fact} - you may be missing leads. I can fix the post around {offer} now. Reply YES."
        cta = "Reply YES"
    elif route == "perf_spike":
        body = f"{name}, {fact}. 3 merchants nearby used a quick follow-up after similar spikes. Want me to draft one for {offer}?"
        cta = "Say GO"
    elif route == "milestone":
        body = f"{name}, {fact}. Strong trust signal - want me to draft a thank-you Google post today?"
        cta = "Reply YES"
    elif route == "festival":
        festival_fact = _first_fact_with(["festival"], facts) or fact
        body = f"{name}, {festival_fact}. {offer} fits the occasion. I can draft a campaign now. Reply YES or STOP."
        cta = "YES or STOP"
    elif route == "reactivation":
        body = f"{name}, {fact}. I can turn this into a useful Google post in 5 min."
        cta = "Say GO"
    elif route == "review_insight":
        review_fact = _first_fact_with(["review"], facts) or fact
        body = f"{name}, {review_fact}. Is this a real pain point? I can draft a calm reply template."
        cta = "Reply YES"
    elif route == "competitive":
        comp_fact = _first_fact_with(["competitor"], facts) or fact
        body = f"{name}, {comp_fact}. Your {offer} can stand out here. Want to see how you compare?"
        cta = "Reply COMPARE"
    elif route == "curious_ask":
        business = _deep_get(merchant, "business_name", "identity.business_name", "name") or "your store"
        body = f"{name}, quick ask: what service got the most enquiries this week at {business}? I'll turn it into a Google post + WhatsApp reply."
        cta = "Reply with one service"
    elif route == "content_nudge":
        body = f"{name}, {fact}. I'll write a fresh post around {offer} - just say GO."
        cta = "Say GO"
    else:
        body = f"{name}, {fact}. Want me to draft the next WhatsApp or Google post for {offer}?"
        cta = "Reply YES"

    if hinglish and route != "recall" and not re.search(r"\b(haan|karo|main|aap|hai|hain)\b", body.lower()):
        body = body.rstrip(".") + " - main draft kar doon?"

    send_as = "merchant_on_behalf" if route == "recall" else "vera"
    return {
        "body": _body_with_limit(body),
        "cta": cta,
        "send_as": send_as,
        "suppression_key": "",
        "rationale": f"Fallback composition used route={route} and fact='{fact}'.",
    }


def _suppression_key(
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None,
    route: str,
) -> str:
    explicit = trigger.get("suppression_key") if isinstance(trigger, dict) else None
    if explicit:
        return str(explicit)
    merchant_id = (
        trigger.get("merchant_id")
        or _deep_get(merchant, "merchant_id", "id", "identity.merchant_id")
        or "merchant"
    )
    customer_id = (
        trigger.get("customer_id")
        or _deep_get(customer, "customer_id", "id", "profile.customer_id")
        or "merchant"
    )
    trigger_id = trigger.get("trigger_id") or trigger.get("id")
    if not trigger_id:
        stable_trigger = _stable_trigger_for_hash(trigger)
        encoded_trigger = json.dumps(
            stable_trigger,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        trigger_id = "anon-" + hashlib.sha1(encoded_trigger.encode("utf-8")).hexdigest()[:12]
    date_key = trigger.get("suppression_date") or trigger.get("date") or trigger.get("scheduled_for") or ""
    raw = f"{merchant_id}:{customer_id}:{trigger_id}:{route}:{date_key}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"vera:{merchant_id}:{customer_id}:{route}:{digest}"


def _stable_trigger_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        stable: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_l = key_text.lower()
            if any(field in key_l for field in _VOLATILE_TRIGGER_HASH_FIELDS):
                continue
            stable[key_text] = _stable_trigger_for_hash(item)
        return stable
    if isinstance(value, list):
        return [_stable_trigger_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_trigger_for_hash(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    category = category or {}
    merchant = merchant or {}
    trigger = trigger or {}
    route = route_for_trigger(trigger)
    key_facts = extract_key_facts(category, merchant, trigger, customer)
    canonical_key = _suppression_key(merchant, trigger, customer, route)
    system_prompt, user_prompt = build_prompts(route, category, merchant, trigger, customer, key_facts)
    fallback_result = _fallback_message(route, category, merchant, trigger, customer, key_facts)
    fallback_result["suppression_key"] = canonical_key
    safe_fallback_result = fallback_result if not _validation_errors(fallback_result, category) else None

    try:
        result = _chat_json(COMPOSE_MODEL, system_prompt, user_prompt)
    except Exception as exc:
        result = dict(fallback_result)
        result["rationale"] += f" OpenAI unavailable or failed: {type(exc).__name__}."

    result["_system_prompt"] = system_prompt
    result["_user_prompt"] = user_prompt
    result["_route"] = route
    result["_key_facts"] = key_facts
    result["_merchant_name"] = _merchant_name(merchant)
    if not result.get("suppression_key"):
        result["suppression_key"] = canonical_key

    validated = validate_and_fix(result, category, trigger, fallback_result=safe_fallback_result)
    if not validated.get("suppression_key"):
        validated["suppression_key"] = canonical_key
    validated["route"] = route
    validated["key_facts"] = key_facts
    return validated
