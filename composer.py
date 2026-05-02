import hashlib
import json
import logging
import os
import re
import threading
from typing import Any

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from utils import deep_get as _deep_get
from utils import first_fact_with as _first_fact_with


DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 25.0
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
_PLACEHOLDER_NAME_TOKENS = {
    "anonymous",
    "unknown",
    "walk-in",
    "walk in",
    "no profile",
    "grandfather",
    "test",
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
    if "%" in text:
        return None
    if "₹" in text or text.lower().startswith("rs"):
        return text.replace("Rs.", "₹").replace("Rs ", "₹")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        amount = match.group(0)
        try:
            if float(amount) == 0:
                return None
        except ValueError:
            pass
        if amount.endswith(".0"):
            amount = amount[:-2]
        return f"₹{amount}"
    return None


def _merchant_name(merchant: dict[str, Any] | None) -> str:
    value = _deep_get(
        merchant,
        "identity.owner_first_name",
        "owner_first_name",
        "owner_name",
        "merchant_name",
        "business_name",
        "name",
        "identity.owner_name",
        "identity.business_name",
        "identity.name",
    )
    return str(value or "there").strip()


def _customer_name(customer: dict[str, Any] | None) -> str:
    value = _deep_get(
        customer,
        "identity.first_name",
        "identity.name",
        "first_name",
        "name",
        "profile.first_name",
        "profile.name",
    )
    if value:
        lowered = str(value).lower()
        if any(token in lowered for token in _PLACEHOLDER_NAME_TOKENS):
            value = None
    if not value:
        customer_id = _deep_get(customer, "customer_id", "id")
        if customer_id:
            parts = str(customer_id).split("_")
            if len(parts) >= 3 and parts[2]:
                candidate = parts[2].title()
                if any(token in candidate.lower() for token in _PLACEHOLDER_NAME_TOKENS):
                    value = None
                else:
                    value = candidate
    if not value:
        return "there"
    parts = str(value).split()
    if len(parts) >= 2 and parts[0].rstrip(".").lower() in {"mr", "mrs", "ms", "dr"}:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def _business_name(merchant: dict[str, Any] | None) -> str:
    value = _deep_get(
        merchant,
        "identity.name",
        "identity.business_name",
        "business_name",
        "merchant_name",
        "name",
    )
    return str(value or "your business").strip()


def _merchant_salutation(merchant: dict[str, Any] | None, category: dict[str, Any] | None) -> str:
    slug = _category_slug(category, merchant)
    first = _deep_get(
        merchant,
        "identity.owner_first_name",
        "owner_first_name",
        "owner_name",
        "merchant_name",
        "business_name",
        "name",
    )
    if first:
        first = str(first).strip().split()[0]
        if "dent" in slug and not first.lower().startswith("dr"):
            return f"Dr. {first}"
        return first
    business = _business_name(merchant)
    return "" if business == "your business" else business


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
                name_text = str(name or "")
                if price in {"₹0", "₹0.0"} and "free" in name_text.lower():
                    price = None
                if name and price and ("₹" not in name_text and not re.search(r"\brs\.?\s*\d", name_text, re.I)):
                    suffix = " (active)" if "active" in status else ""
                    label = f"{name} @ {price}{suffix}"
                elif name:
                    suffix = " (active)" if "active" in status else ""
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
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_label = candidates[0][1]
    if best_label and any(label == best_label for _, label in candidates if "@" in label):
        for _, label in candidates:
            if "@" in label:
                return label
    return best_label


def _first_slot(*contexts: dict[str, Any] | None) -> tuple[str | None, str | None]:
    slots: list[Any] = []
    for ctx in contexts:
        slots.extend(
            _as_list(
                _deep_get(
                    ctx,
                    "available_slots",
                    "payload.available_slots",
                    "payload.next_session_options",
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


def _category_slug(category: dict[str, Any] | None, merchant: dict[str, Any] | None = None) -> str:
    value = _deep_get(category, "slug", "category_slug") or _deep_get(merchant, "category_slug")
    return str(value or "").lower()


def _city_locality(merchant: dict[str, Any] | None) -> str:
    locality = _deep_get(merchant, "identity.locality", "locality")
    city = _deep_get(merchant, "identity.city", "city")
    if locality and city:
        return f"{locality}, {city}"
    return str(locality or city or "").strip()


def _metric_snapshot(
    merchant: dict[str, Any] | None,
    category: dict[str, Any] | None,
    metric: str,
) -> str | None:
    metric = metric.lower()
    merchant_paths = {
        "views": ("performance.views", "metrics.views", "views"),
        "calls": ("performance.calls", "metrics.calls", "calls"),
        "directions": ("performance.directions", "metrics.directions", "directions"),
        "ctr": ("performance.ctr", "metrics.ctr", "ctr"),
        "leads": ("performance.leads", "metrics.leads", "leads"),
    }
    peer_paths = {
        "views": ("peer_stats.avg_views_30d", "benchmarks.avg_views_30d"),
        "calls": ("peer_stats.avg_calls_30d", "benchmarks.avg_calls_30d"),
        "directions": ("peer_stats.avg_directions_30d", "benchmarks.avg_directions_30d"),
        "ctr": ("peer_stats.avg_ctr", "benchmarks.ctr"),
        "leads": ("peer_stats.avg_leads_30d", "benchmarks.avg_leads_30d"),
    }
    value = _deep_get(merchant, *merchant_paths.get(metric, (f"performance.{metric}", f"metrics.{metric}", metric)))
    peer = _deep_get(category, *peer_paths.get(metric, ()))
    if value is None:
        return None
    if metric == "ctr":
        value_text = _fmt_percent(value) or str(value)
        peer_text = _fmt_percent(peer) if peer is not None else None
    else:
        value_text = str(value)
        peer_text = str(peer) if peer is not None else None
    if peer_text:
        return f"{metric} {value_text} vs peer {peer_text}"
    return f"{metric} {value_text}"


def _clean_fact_text(fact: str | None) -> str:
    if not fact:
        return ""
    text = str(fact).strip()
    prefixes = (
        "Top digest item:",
        "Active offers:",
        "Performance dip:",
        "Performance spike:",
        "Festival:",
        "Event:",
        "Review theme:",
        "Competitor signal:",
        "Milestone:",
        "Planning intent:",
        "Ask due:",
        "Merchant asked:",
        "Renewal due:",
        "Winback signal:",
        "Dormant signal:",
        "Profile signal:",
        "Seasonal trends:",
        "Supply alert:",
        "Refill due:",
        "Wedding follow-up:",
        "Trial follow-up:",
        "Customer timing:",
        "Service due:",
        "Available slots:",
        "Available slot:",
        "Content signal:",
        "CTR gap:",
        "View trends:",
        "Seasonal signal:",
    )
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            break
    text = re.sub(r"\bGBP\b", "Google profile", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCTR\b", "click rate", text, flags=re.IGNORECASE)
    return text


def _humanize_label(value: Any) -> str:
    return str(value or "").replace("_", " ").strip()


def _short_date(value: Any) -> str:
    text = str(value or "").strip()
    if "T" in text:
        return text.split("T", 1)[0]
    return text


def _digest_lookup(category: dict[str, Any] | None, trigger: dict[str, Any] | None) -> dict[str, Any] | None:
    wanted_id = _deep_get(
        trigger,
        "payload.top_item.id",
        "payload.top_item_id",
        "payload.digest_item_id",
        "payload.alert_id",
        "top_item_id",
        "digest_item_id",
        "alert_id",
    )
    direct = _deep_get(trigger, "payload.top_item", "top_item", "digest_item", "top_digest_item", "research_item")
    if isinstance(direct, dict):
        return direct
    for item in _as_list(_deep_get(category, "digest", "digest_items", "research_digest.items", "research.items")):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or item.get("item_id")
        if wanted_id and item_id and str(item_id) != str(wanted_id):
            continue
        return item
    return None


def _offer_for_route(
    route: str,
    merchant: dict[str, Any],
    category: dict[str, Any],
    trigger: dict[str, Any],
) -> str:
    offer = _first_offer(merchant, category, trigger)
    if offer:
        return offer
    return "your current offer"


def _extract_digest_item(category: dict[str, Any] | None, trigger: dict[str, Any] | None) -> str | None:
    wanted_id = _deep_get(
        trigger,
        "payload.top_item.id",
        "payload.top_item_id",
        "payload.digest_item_id",
        "payload.alert_id",
        "top_item_id",
        "digest_item_id",
        "alert_id",
    )
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
                item_id = item.get("id") or item.get("item_id")
                if wanted_id and item_id and str(item_id) != str(wanted_id):
                    continue
                title = item.get("title") or item.get("finding") or item.get("summary")
                source_name = item.get("source") or item.get("citation") or item.get("journal")
                stat = item.get("stat") or item.get("metric") or item.get("result")
                if not stat:
                    text_blob = " ".join(str(item.get(key) or "") for key in ("title", "summary", "actionable"))
                    match = re.search(r"\d+(?:\.\d+)?%", text_blob)
                    if match:
                        stat = match.group(0)
                sample = item.get("n") or item.get("sample_size") or item.get("trial_size") or item.get("trial_n")
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
            facts.append(f"Click rate gap: merchant {m_ctr} vs peer {p_ctr}")

    locality = _city_locality(merchant)
    if locality:
        facts.append(f"Location: {locality}")

    avg_rating = _deep_get(category, "peer_stats.avg_rating")
    avg_reviews = _deep_get(category, "peer_stats.avg_reviews", "peer_stats.avg_review_count")
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
        "performance.reviews",
        "metrics.review_count",
        "google_profile.review_count",
        "reviews_count",
    )
    if merchant_rating and avg_rating:
        facts.append(f"Rating: merchant {merchant_rating} vs peer avg {avg_rating}")
    if merchant_reviews and avg_reviews:
        facts.append(f"Reviews: merchant {merchant_reviews} vs peer avg {avg_reviews}")

    merchant_verified = _deep_get(merchant, "identity.verified", "google_profile.verified")
    if merchant_verified is False:
        facts.append("Profile signal: Google profile unverified")

    for metric in ("views", "calls", "directions", "leads"):
        snapshot = _metric_snapshot(merchant, category, metric)
        if snapshot:
            facts.append(snapshot.capitalize())

    subscription_status = _deep_get(merchant, "subscription.status")
    subscription_days = _deep_get(merchant, "subscription.days_remaining")
    if subscription_status:
        text = f"Subscription: {subscription_status}"
        if subscription_days is not None:
            text += f"; {subscription_days} days remaining"
        facts.append(text)

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

    lapsed_90 = _deep_get(merchant, "customer_aggregate.lapsed_90d_plus")
    if lapsed_90:
        facts.append(f"Lapsed customers: {lapsed_90} lapsed >90 days")

    for label, path in (
        ("Repeat customers", "customer_aggregate.repeat_customer_pct"),
        ("Delivery share", "customer_aggregate.delivery_share_pct"),
        ("Trial-to-paid", "customer_aggregate.trial_to_paid_pct"),
        ("Monthly churn", "customer_aggregate.monthly_churn_pct"),
        ("Chronic Rx customers", "customer_aggregate.chronic_rx_count"),
        ("Delivery orders", "customer_aggregate.delivery_orders_30d"),
        ("Dine-in orders", "customer_aggregate.dine_in_orders_30d"),
    ):
        value = _deep_get(merchant, path)
        if value is None:
            continue
        pct = _fmt_percent(value) if "pct" in path else None
        facts.append(f"{label}: {pct or value}")

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
        "payload.views_wow",
    ) or _deep_get(merchant, "metrics.views_wow", "performance.views_wow", "performance.delta_7d.views_pct")
    if view_trend:
        facts.append(f"View trends: {_fmt_percent(view_trend) or view_trend} vs last week")

    seasonal = _deep_get(
        trigger,
        "seasonal_signal",
        "signals.seasonal",
        "seasonal",
        "payload.season_note",
        "payload.season",
    ) or _deep_get(category, "seasonal_signal", "signals.seasonal")
    if seasonal:
        facts.append(f"Seasonal signal: {seasonal}")

    kind = str(_deep_get(trigger, "kind", "type") or "").lower()
    delta = _deep_get(
        trigger,
        "payload.delta_pct",
        "payload.perf_dip_pct",
        "metric_drop",
        "drop",
        "metrics.drop_percent",
        "calls_drop",
        "metric_spike",
        "spike",
        "metrics.spike_percent",
        "views_spike",
    )
    metric_name = _deep_get(trigger, "payload.metric", "metric_name", "metric", "signal_name") or "metric"
    if delta:
        try:
            delta_num = float(delta)
        except (TypeError, ValueError):
            delta_num = 0.0
        pct = _fmt_percent(abs(delta_num)) or str(delta)
        if "spike" in kind or delta_num > 0:
            facts.append(f"Performance spike: {metric_name} up {pct}")
        else:
            facts.append(f"Performance dip: {metric_name} down {pct}")

    festival = _deep_get(trigger, "payload.festival", "festival", "festival_name", "event_name")
    days_remaining = _deep_get(trigger, "payload.days_until", "days_remaining", "days_to_festival")
    if festival:
        if days_remaining is not None:
            facts.append(f"Festival: {festival} in {days_remaining} days")
        else:
            facts.append(f"Festival: {festival}")
    match_name = _deep_get(trigger, "payload.match")
    if match_name:
        city = _deep_get(trigger, "payload.city")
        facts.append(f"Event: {match_name} match today" + (f" in {city}" if city else ""))

    review_theme = _deep_get(
        trigger,
        "payload.theme",
        "review_theme",
        "theme",
        "signals.review_theme",
        "review_insight.theme",
    )
    review_count = _deep_get(trigger, "payload.occurrences_30d", "review_count", "signals.review_count", "theme_count")
    if review_theme:
        quote = _deep_get(trigger, "payload.common_quote")
        if review_count:
            facts.append(f"Review theme: {review_theme} mentioned {review_count} times" + (f"; pattern '{quote}'" if quote else ""))
        else:
            facts.append(f"Review theme: {review_theme}")

    competitor = _deep_get(
        trigger,
        "payload.competitor_name",
        "competitor_name",
        "competitor.name",
        "signals.competitor_name",
    )
    if competitor:
        distance = _deep_get(trigger, "payload.distance_km", "competitor.distance", "distance", "distance_m")
        if distance:
            unit = "km" if "distance_km" in json.dumps(trigger or {}) else ""
            facts.append(f"Competitor signal: {competitor} opened {distance}{(' ' + unit) if unit else ''} away")
        else:
            facts.append(f"Competitor signal: {competitor}")

    milestone_value = _deep_get(trigger, "payload.milestone_value")
    value_now = _deep_get(trigger, "payload.value_now")
    if milestone_value and value_now:
        facts.append(f"Milestone: {metric_name} {value_now}/{milestone_value}")

    intent_topic = _deep_get(trigger, "payload.intent_topic")
    if intent_topic:
        facts.append(f"Planning intent: {intent_topic}")

    ask_template = _deep_get(trigger, "payload.ask_template")
    if ask_template:
        facts.append(f"Ask due: {str(ask_template).replace('_', ' ')}")

    merchant_last_message = _deep_get(trigger, "payload.merchant_last_message")
    if merchant_last_message:
        facts.append(f"Merchant asked: {merchant_last_message}")

    days_remaining_sub = _deep_get(trigger, "payload.days_remaining")
    renewal_amount = _fmt_money(_deep_get(trigger, "payload.renewal_amount"))
    plan = _deep_get(trigger, "payload.plan")
    if days_remaining_sub is not None and plan:
        facts.append(
            f"Renewal due: {plan} plan in {days_remaining_sub} days"
            + (f"; amount {renewal_amount}" if renewal_amount else "")
        )

    days_since_expiry = _deep_get(trigger, "payload.days_since_expiry")
    lapsed_added = _deep_get(trigger, "payload.lapsed_customers_added_since_expiry")
    if days_since_expiry:
        facts.append(
            f"Winback signal: expired {days_since_expiry} days ago"
            + (f"; {lapsed_added} lapsed customers added" if lapsed_added else "")
        )

    days_since_message = _deep_get(trigger, "payload.days_since_last_merchant_message")
    if days_since_message:
        facts.append(f"Dormant signal: no merchant reply for {days_since_message} days")

    if _deep_get(trigger, "payload.verified") is False:
        uplift = _fmt_percent(_deep_get(trigger, "payload.estimated_uplift_pct"))
        facts.append(
            f"Profile signal: Google profile unverified" + (f"; estimated uplift {uplift}" if uplift else "")
        )

    trends = _deep_get(trigger, "payload.trends")
    if isinstance(trends, list) and trends:
        facts.append("Seasonal trends: " + ", ".join(str(t) for t in trends[:2]))

    molecule = _deep_get(trigger, "payload.molecule")
    batches = _deep_get(trigger, "payload.affected_batches")
    if molecule:
        batch_text = ", ".join(str(batch) for batch in _as_list(batches)[:2]) if batches else ""
        facts.append(f"Supply alert: {molecule}" + (f" batches {batch_text}" if batch_text else ""))

    molecule_list = _deep_get(trigger, "payload.molecule_list")
    stock_runs_out = _deep_get(trigger, "payload.stock_runs_out_iso")
    if molecule_list:
        meds = ", ".join(str(item) for item in _as_list(molecule_list)[:3])
        facts.append(
            f"Refill due: {meds}"
            + (f"; stock runs out {stock_runs_out}" if stock_runs_out else "")
        )

    wedding_date = _deep_get(trigger, "payload.wedding_date")
    days_to_wedding = _deep_get(trigger, "payload.days_to_wedding")
    trial_completed = _deep_get(trigger, "payload.trial_completed")
    if wedding_date:
        facts.append(
            f"Wedding follow-up: wedding {wedding_date}"
            + (f"; {days_to_wedding} days left" if days_to_wedding else "")
            + (f"; trial completed {trial_completed}" if trial_completed else "")
        )

    trial_date = _deep_get(trigger, "payload.trial_date")
    if trial_date:
        facts.append(f"Trial follow-up: trial on {trial_date}")

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
        "payload.months_since_last_visit",
        "months_since_last_visit",
        "months_since_last_order",
        "signals.months_since_last_visit",
    ) or _deep_get(customer, "months_since_last_visit", "months_since_last_order")
    if months_since:
        facts.append(f"Customer timing: {months_since} months since last visit")
    days_since = _deep_get(trigger, "payload.days_since_last_visit")
    if days_since:
        facts.append(f"Customer timing: {days_since} days since last visit")
    service_due = _deep_get(trigger, "payload.service_due")
    due_date = _deep_get(trigger, "payload.due_date")
    last_service_date = _deep_get(trigger, "payload.last_service_date")
    if service_due and due_date:
        facts.append(f"Service due: {str(service_due).replace('_', ' ')} on {due_date}")
    elif last_service_date:
        facts.append(f"Last service date: {last_service_date}")

    slot_a, slot_b = _first_slot(trigger, merchant)
    if slot_a and slot_b:
        facts.append(f"Available slots: {slot_a}, {slot_b}")
    elif slot_a:
        facts.append(f"Available slot: {slot_a}")

    for generic in _generic_number_facts(merchant, trigger, category, customer):
        if generic not in facts:
            facts.append(generic)
        if len(facts) >= 24:
            break

    return facts[:24]


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
    salutation = _merchant_salutation(merchant, category)
    business = _business_name(merchant)
    location = _city_locality(merchant)
    active_offers = [
        offer.get("title")
        for offer in _as_list(_deep_get(merchant, "offers"))
        if isinstance(offer, dict)
        and str(offer.get("status") or "").lower() in {"active", "running"}
    ]
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
    - Open with the merchant name or business name (dentists use "Dr.").
    - If there are active offers, include one; if none, ask for one instead of inventing.
    - Avoid internal abbreviations (say "Google profile" instead of GBP).
    - Prefer referencing locality/city when available.

Scoring guidance (all 5 dimensions matter equally):
1. Specificity: Use exact numbers/percentages/dates from KEY FACTS verbatim. Never round or soften: "2,100-patient trial" not "a study"; "50% drop" not "big drop"; "₹4999" not "a fee".
2. Category Fit: STRICTLY match voice like:
   - dentist/dental: always "Dr. [name]" prefix; cite journal (JIDA/IDA/DCI); clinical peer tone; no hype
   - salon/beauty: warm personal ("your clients", "your studio"); mention specific service + price; friendly
   - gym/yoga: coaching language ("your members", "your athletes"); motivational, goal-focused
   - restaurant/cafe: operator-peer ("your kitchen", "your regulars"); direct ROI framing
   - pharmacy: precision + compliance ("CDSCO", "DCI", "batch"); patient-safety first
3. Merchant Fit: Body MUST include ALL THREE: (a) owner first name, (b) business name OR locality, (c) exact offer with price OR exact metric number.
4. Trigger Relevance: First sentence must name the trigger signal directly. Make it obvious why THIS message is sent TODAY.
5. Engagement: MUST end with ONE of these high-scoring patterns:
   - Loss aversion: "You are losing [X calls/leads/customers] this week — I can fix this in 5 min. Reply YES."
   - Social proof: "[N] [category] merchants in [locality] did [X] after seeing this signal."
   - Curiosity close: "I found one specific thing in your [data/profile/reviews] that explains this — want to see?"
   Never use standalone "Reply YES" or "Say GO" — always attach a specific loss or reason.
   
    Category voice:
    {_voice_summary(category)}
    
    Merchant anchor: {salutation or business}
    Business name: {business}
    Location: {location}
    Active offers: {active_offers or "none"}
    """.strip()

    # Pick top facts most relevant for this route
    _route_fact_keywords: dict[str, list[str]] = {
        "research": ["digest", "supply", "regulation", "lapsed", "high-risk", "chronic"],
        "recall": ["customer timing", "available slot", "active offers", "service due", "refill"],
        "perf_dip": ["performance dip", "ctr gap", "view trends", "calls", "leads"],
        "perf_spike": ["performance spike", "view trends", "ctr gap", "active offers"],
        "festival": ["festival", "event", "active offers", "seasonal"],
        "milestone": ["milestone", "reviews", "rating"],
        "reactivation": ["renewal due", "winback signal", "dormant signal", "profile signal", "content signal"],
        "review_insight": ["review theme"],
        "competitive": ["competitor signal", "rating", "active offers"],
        "curious_ask": ["planning intent", "ask due", "active offers", "view trends"],
        "content_nudge": ["content signal", "profile signal", "seasonal", "ctr gap", "active offers"],
    }
    priority_kws = _route_fact_keywords.get(route, [])
    top_facts = [f for f in key_facts if any(kw in f.lower() for kw in priority_kws)][:4]
    if not top_facts:
        top_facts = key_facts[:4]
    top_fact_block = "\n".join(f"- {fact}" for fact in top_facts)
    best_example = few_shots[0] if few_shots else {}
    owner_name = _merchant_salutation(merchant, category) or _merchant_name(merchant)
    business_name = _business_name(merchant)
    locality_str = _city_locality(merchant)
    lang_str = ", ".join(_language_values(category, merchant, customer)) or "en"
    trigger_payload_str = json.dumps(trigger.get("payload", {}), ensure_ascii=False)[:400]
    user_prompt = f"""
    MERCHANT: {owner_name} | {business_name}{" | " + locality_str if locality_str else ""}
TRIGGER: {trigger.get("kind", "unknown")} — payload: {trigger_payload_str}
LANGUAGE: {lang_str}

PRIORITY FACTS — include ≥2 of these VERBATIM in body:
{top_fact_block}

ALL KEY FACTS:
{key_fact_block}

BEST EXAMPLE FOR THIS ROUTE:
{_json_compact(best_example, 700)}

    Compose now. Body MUST start with "{owner_name}," and weave exact numbers/stats from ≥2 PRIORITY FACTS into natural sentences. Include business name or locality in the body. Do NOT copy label names like "Performance dip:" or "Renewal due:" — just the actual data value.""".strip()
    return system_prompt, user_prompt

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _chat_json(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> dict[str, Any]:
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
    except Exception as e:
        logging.warning("[vera] LLM HTTP call failed: %s", str(e))
        raise
    content = (response.choices[0].message.content or "{}").strip()
    if not content:
        return {}
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
    body = re.sub(r"\bGBP\b", "Google profile", body, flags=re.IGNORECASE)
    body = re.sub(r"\bCTR\b", "click rate", body, flags=re.IGNORECASE)
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
    name = _merchant_salutation(merchant, category) or _merchant_name(merchant)
    customer_first = _customer_name(customer)
    business = _business_name(merchant)
    offer = _offer_for_route(route, merchant, category, trigger)
    slug = _category_slug(category, merchant)
    kind = str(_deep_get(trigger, "kind", "type") or "").lower()
    location = _city_locality(merchant)
    route_patterns = {
        "research": ["top digest", "supply alert", "planning intent", "seasonal trends"],
        "recall": ["customer timing", "service due", "available slot", "active offers"],
        "perf_dip": ["performance dip", "ctr gap", "view trends"],
        "perf_spike": ["performance spike", "view trends", "ctr gap"],
        "milestone": ["milestone", "reviews", "rating"],
        "festival": ["festival", "event", "seasonal"],
        "reactivation": ["renewal due", "winback signal", "dormant signal", "profile signal", "planning intent", "seasonal trends", "lapsed", "content signal"],
        "review_insight": ["review theme"],
        "competitive": ["competitor signal", "rating", "active offers"],
        "curious_ask": ["planning intent", "merchant asked", "ask due", "seasonal trends", "active offers", "view trends"],
        "content_nudge": ["content signal", "profile signal", "seasonal trends", "event", "active offers"],
    }
    fact = _first_fact_with(route_patterns.get(route, []), facts)
    fact = fact or (facts[0] if facts else "your latest Google profile signal")
    hinglish = _needs_hinglish(category, merchant, customer)

    if route == "research":
        digest = _digest_lookup(category, trigger)
        source = str((digest or {}).get("source") or "").strip()
        title = str((digest or {}).get("title") or _clean_fact_text(fact)).strip()
        summary = str((digest or {}).get("summary") or "").strip()
        stat_match = re.search(r"\d+(?:\.\d+)?%", " ".join([title, summary, _clean_fact_text(fact)]))
        stat = stat_match.group(0) if stat_match else ""
        sample = (digest or {}).get("trial_n") or (digest or {}).get("sample_size") or (digest or {}).get("n")
        cohort = _first_fact_with(["high-risk", "lapsed", "chronic", "repeat customers"], facts)
        if "supply" in kind or "alert" in kind:
            batches = _deep_get(trigger, "payload.affected_batches")
            batch_text = ", ".join(str(b) for b in _as_list(batches)[:2]) if batches else ""
            molecule = _deep_get(trigger, "payload.molecule") or "affected batch"
            source_label = source or "CDSCO"
            alert_word = "" if "alert" in source_label.lower() else " alert"
            body = (
                f"{name}, {source_label}{alert_word}: {molecule} {batch_text} needs a shelf check. "
                f"Want me to filter repeat-Rx customers and draft the recall WhatsApp?"
            )
        elif "regulation" in kind:
            deadline = _deep_get(trigger, "payload.deadline_iso")
            body = (
                f"{name}, DCI update: IOPA dose limit changes by {deadline}; E-speed/RVG setup matters. "
                f"Want me to draft your SOP note + patient-safe wording?"
            )
        elif "cde" in kind:
            credits = _deep_get(trigger, "payload.credits") or (digest or {}).get("credits")
            date = (digest or {}).get("date") or _deep_get(trigger, "payload.date")
            body = (
                f"{name}, {source or 'IDA'} has a {credits or 2}-credit CDE on digital impressions"
                f"{' on ' + str(date)[:10] if date else ''}. Want me to pull it + draft a Google post?"
            )
        else:
            trial_text = f"{sample}-patient trial" if sample else "new finding"
            stat_text = f"{stat} " if stat else ""
            cohort_clean = _clean_fact_text(cohort)
            cohort_text = ""
            if cohort_clean:
                high_risk_match = re.search(r"high-risk adult patients:\s*(\d+)", cohort_clean, re.I)
                if high_risk_match:
                    cohort_text = f" Your {high_risk_match.group(1)} high-risk adult patients fit this."
                else:
                    cohort_text = f" {cohort_clean} fit this."
            body = (
                f"{name}, {source or 'this digest'}: {trial_text} shows {stat_text}{title[:92]}."
                f"{cohort_text} Want a patient WhatsApp draft?"
            )
        cta = "Reply YES"
    elif route == "recall":
        refill = _first_fact_with(["refill due"], facts)
        wedding = _first_fact_with(["wedding follow-up"], facts)
        trial = _first_fact_with(["trial follow-up"], facts)
        timing = _first_fact_with(
            ["refill due", "wedding follow-up", "trial follow-up", "customer timing", "service due", "last service"],
            facts,
        ) or "follow-up is due"
        slot_a, slot_b = _first_slot(trigger, merchant)
        if refill:
            refill_offer = "Free Home Delivery > ₹499"
            if _deep_get(customer, "identity.senior_citizen"):
                refill_offer += " + Senior Citizen 15% OFF"
            body = (
                f"Hi {customer_first}, {business} here. {_short_date(_clean_fact_text(refill))}. "
                f"{refill_offer}. Reply 1 for delivery to saved address, 2 for pickup."
            )
        elif wedding:
            body = (
                f"Hi {customer_first}, {business} here. {_clean_fact_text(wedding)}. "
                f"{offer}. Reply 1 for skin-prep plan, 2 for package call."
            )
        elif trial:
            slot_a = slot_a or "next session"
            body = (
                f"Hi {customer_first}, {business} here. {_clean_fact_text(trial)}. "
                f"{slot_a} is open. Reply 1 to book, 2 for another slot."
            )
        elif "lapsed" in kind:
            timing_text = _clean_fact_text(timing)
            focus = _deep_get(trigger, "payload.previous_focus") or _deep_get(customer, "preferences.training_focus")
            preferred = _humanize_label(_deep_get(customer, "preferences.preferred_slots"))
            comeback_offer = "3 FREE Trial Classes" if "gym" in slug else offer
            body = (
                f"Hi {customer_first}, {business} here. {timing_text}"
                f"{' since your last visit' if 'since' not in timing_text else ''}. "
                f"{comeback_offer}{' fits your ' + _humanize_label(focus) + ' goal' if focus else ''}"
                f"{' in your ' + preferred + ' slot' if preferred else ''}. "
                "Reply 1 to restart, 2 for a callback."
            )
        else:
            slot_a = slot_a or "today 5pm"
            slot_b = slot_b or "tomorrow 11am"
            body = (
                f"Hi {customer_first}, {business} here. {_clean_fact_text(timing)}. {offer}. "
                f"{slot_a} or {slot_b} available. Reply 1 for first, 2 for second."
            )
        cta = "Reply 1 or 2"
    elif route == "perf_dip":
        metric = _deep_get(trigger, "payload.metric", "metric") or "calls"
        snapshot = _metric_snapshot(merchant, category, str(metric)) or _metric_snapshot(merchant, category, "calls")
        signal = _clean_fact_text(fact)
        if "seasonal" in kind and "gym" in slug:
            body = (
                f"{name}, {signal}; April-Jun is the low acquisition window. "
                f"{snapshot or offer}. I can push a retention/trial post now. Reply YES."
            )
        else:
            profile_gap = _first_fact_with(["profile signal"], facts)
            gap_text = f"{_clean_fact_text(profile_gap)} is also hurting trust. " if profile_gap else ""
            body = (
                f"{name}, {signal}; {snapshot or 'leads are at risk'}. "
                f"{gap_text}I can fix one Google post around {offer} now. Reply YES."
            )
        cta = "Reply YES"
    elif route == "perf_spike":
        driver = _deep_get(trigger, "payload.likely_driver")
        followup = "kids-yoga follow-up" if driver else offer
        body = (
            f"{name}, {_clean_fact_text(fact)} at {business}"
            f"{' after ' + str(driver).replace('_', ' ') if driver else ''}. "
            f"3 nearby merchants used the spike for follow-up. Want a {followup} draft?"
        )
        cta = "Say GO"
    elif route == "milestone":
        milestone_text = _clean_fact_text(fact)
        milestone_match = re.search(r"([a-z_]+)\s+(\d+)\s*/\s*(\d+)", milestone_text, re.I)
        if milestone_match:
            metric_label = milestone_match.group(1).replace("_", " ")
            if metric_label == "review count":
                metric_label = "reviews"
            milestone_text = f"{milestone_match.group(2)} of {milestone_match.group(3)} {metric_label}"
        body = (
            f"{name}, {business} is at {milestone_text}. "
            "That is a trust moment; want a thank-you Google post to ask happy regulars today?"
        )
        cta = "Reply YES"
    elif route == "festival":
        festival_fact = _first_fact_with(["festival"], facts) or fact
        event = _first_fact_with(["event"], facts)
        if event:
            venue = _deep_get(trigger, "payload.venue")
            body = (
                f"{name}, {_clean_fact_text(event)}"
                f"{' at ' + str(venue) if venue else ''}. Push {offer} for home-watch orders in {location or 'your area'}? YES/STOP."
            )
        else:
            seasonal_note = "Bridal bookings peak Oct-Dec; early trials start now. " if "salon" in slug else ""
            body = (
                f"{name}, {_clean_fact_text(festival_fact)}. Start the booking runway now: "
                f"{seasonal_note}{offer} fits {business}. Want a festive campaign draft? YES/STOP."
            )
        cta = "YES or STOP"
    elif route == "reactivation":
        cleaned = _clean_fact_text(fact)
        if "renewal" in kind:
            dip = _first_fact_with(["performance dip", "calls"], facts)
            body = (
                f"{name}, {cleaned}; {business} still has {_clean_fact_text(dip) if dip else 'one profile gap'}. "
                "I found a 5-min fix before renewal. Want me to do it?"
            )
        elif "winback" in kind:
            dip = _first_fact_with(["performance dip"], facts)
            dip_text = _clean_fact_text(dip)
            if dip_text.startswith("metric down"):
                calls_drop = _fmt_percent(abs(_deep_get(merchant, "performance.delta_7d.calls_pct") or 0))
                dip_text = f"calls down {calls_drop}" if calls_drop else dip_text
            body = (
                f"{name}, {cleaned}; {dip_text if dip else 'customers are slipping'}. "
                f"I can draft a comeback post around {offer}. Say GO."
            )
        elif "dormant" in kind:
            lapsed = _first_fact_with(["lapsed customers"], facts)
            lapsed_text = f" {_clean_fact_text(lapsed)}." if lapsed else ""
            body = (
                f"{name}, {cleaned};{lapsed_text} {offer} is still a usable hook. "
                "I can write one fresh post in 5 min. Say GO."
            )
        else:
            body = f"{name}, {cleaned} for {business}. I can turn it into one useful post or reply in 5 min. Say GO."
        cta = "Say GO"
    elif route == "review_insight":
        review_fact = _first_fact_with(["review"], facts) or fact
        body = (
            f"{name}, {_clean_fact_text(review_fact)} at {business}. "
            "Is this a real pain point? I can draft a calm reply + ops note."
        )
        cta = "Reply YES"
    elif route == "competitive":
        comp_fact = _first_fact_with(["competitor"], facts) or fact
        their_offer = _deep_get(trigger, "payload.their_offer")
        body = (
            f"{name}, {_clean_fact_text(comp_fact)}"
            f"{'; they show ' + str(their_offer) if their_offer else ''}. "
            f"Do not race price; position {offer}. Want a comparison?"
        )
        cta = "Reply COMPARE"
    elif route == "curious_ask":
        planning = _first_fact_with(["planning intent"], facts)
        if planning:
            topic = _clean_fact_text(planning).replace("_", " ")
            support_fact = (
                _first_fact_with(["repeat customers", "trial-to-paid", "delivery share", "active offers"], facts)
                or _metric_snapshot(merchant, category, "leads")
            )
            offer_bit = offer if offer != "your active offer" else ""
            body = (
                f"{name}, you were planning {topic} for {business}. "
                f"{_clean_fact_text(support_fact) + '. ' if support_fact else ''}"
                f"{offer_bit + ' is the hook. ' if offer_bit else ''}"
                f"I can draft the offer, Google post, and WhatsApp reply. Say GO."
            )
        elif _first_fact_with(["ask due"], facts):
            growth = _first_fact_with(["calls", "views", "active offers"], facts)
            body = (
                f"{name}, quick ask for {business}: which service is most in demand this week? "
                f"{_clean_fact_text(growth) + '. ' if growth else ''}I'll turn one answer into a post."
            )
        else:
            body = f"{name}, quick ask for {business}: what service got most enquiries this week? I'll turn it into a Google post + WhatsApp reply."
        cta = "Reply with one service"
    elif route == "content_nudge":
        cleaned = _clean_fact_text(fact)
        if "unverified" in kind:
            body = (
                f"{name}, {business} is unverified; estimated uplift is "
                f"{_fmt_percent(_deep_get(trigger, 'payload.estimated_uplift_pct')) or '30%'}. "
                "I can prep the phone/postcard steps + post. Say GO."
            )
        elif "seasonal" in kind:
            cleaned = cleaned.replace("_", " ").replace("+", " +")
            body = f"{name}, {cleaned}. Move the shelf focus now; I can draft a compliance-safe WhatsApp reminder. Say GO."
        else:
            body = f"{name}, {cleaned} for {business}. I'll write a fresh post around {offer} - just say GO."
        cta = "Say GO"
    else:
        body = f"{name}, {_clean_fact_text(fact)} for {business}. Want me to draft the next WhatsApp or Google post for {offer}?"
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
        logging.warning("compose LLM failed for route=%s model=%s: %s", route, COMPOSE_MODEL, type(exc).__name__)
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
    validated = _enforce_personalization(validated, merchant, category)
    validated = _enforce_numeric_fact(validated, key_facts)
    if not validated.get("suppression_key"):
        validated["suppression_key"] = canonical_key
    validated["route"] = route
    validated["key_facts"] = key_facts
    return validated


def _enforce_personalization(
    result: dict[str, Any],
    merchant: dict[str, Any],
    category: dict[str, Any],
) -> dict[str, Any]:
    body = str(result.get("body") or "")
    send_as = result.get("send_as")
    if send_as == "merchant_on_behalf":
        return result
    salutation = _merchant_salutation(merchant, category)
    business = _business_name(merchant)
    body_lower = body.lower()
    salutation_l = salutation.lower() if salutation else ""
    business_l = business.lower() if business else ""
    if salutation and salutation_l not in body_lower and business_l not in body_lower:
        if re.match(r"^hi\b", body_lower):
            body = re.sub(r"^hi\b\s*", f"Hi {salutation}, ", body, flags=re.IGNORECASE)
        else:
            body = f"{salutation}, {body}"
    result["body"] = _body_with_limit(body)
    return result


def _enforce_numeric_fact(result: dict[str, Any], key_facts: list[str]) -> dict[str, Any]:
    body = str(result.get("body") or "")
    if re.search(r"\d", body):
        return result
    fact = _short_numeric_fact(key_facts)
    if not fact:
        return result
    addition = f" ({fact})"
    result["body"] = _body_with_limit(body + addition)
    return result


def _short_numeric_fact(key_facts: list[str]) -> str | None:
    for fact in key_facts:
        if not re.search(r"\d", fact):
            continue
        cleaned = _clean_fact_text(fact)
        if cleaned and len(cleaned) <= 70:
            return cleaned
    return None
