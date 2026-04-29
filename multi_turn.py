import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from composer import (
    _first_offer,
    _json_compact,
    extract_key_facts,
    get_llm_client,
    has_llm_credentials,
    route_for_trigger,
)
from store import ContextStore
from utils import deep_get as _deep_get


CLASSIFY_MODEL = (
    os.getenv("AZURE_OPENAI_CLASSIFY_DEPLOYMENT")
    or os.getenv("OPENAI_CLASSIFY_MODEL")
    or "gpt-4.1-mini"
)
REPLY_MODEL = (
    os.getenv("AZURE_OPENAI_REPLY_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_COMPOSE_DEPLOYMENT")
    or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    or os.getenv("OPENAI_REPLY_MODEL")
    or "gpt-4.1"
)


AUTO_REPLY_SIGNATURES = [
    "aapki jaankari ke liye bahut-bahut shukriya",
    "thank you for contacting",
    "main ek automated assistant hoon",
    "yeh ek automated reply hai",
    "our team will get back",
    "shukriya aapka",
    "hum jald hi aapse sampark karenge",
    "thanks for your message",
    "we will respond as soon as possible",
    "we are currently unavailable",
    "business hours are",
    "please leave your message",
    "this is an automated response",
    "auto reply",
    "automated message",
    "hamari team jaldi reply karegi",
    "kripya apna sandesh chhodein",
    "abhi hum uplabdh nahi hain",
    "we have received your message",
    "humne aapka message receive kar liya hai",
]

HINDI_WORD_RE = re.compile(
    r"\b(hai|hain|haan|ha|nahi|nahin|kya|karo|karna|chahiye|aap|hum|mera|"
    r"mere|abhi|baad|kal|theek|thik|shukriya|dhanyavaad|kripya)\b",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_reply_language(message: str) -> str:
    text = message or ""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if HINDI_WORD_RE.search(text):
        return "hi-en"
    return "en"


def is_auto_reply(message: str) -> bool:
    text = re.sub(r"\s+", " ", (message or "").strip().lower())
    if not text:
        return False
    return any(signature in text for signature in AUTO_REPLY_SIGNATURES)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def _classify_with_llm(message: str, languages: str) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify merchant message intent. Return JSON: {intent: string}. "
                    "Options: accept, decline, join_intent, question, auto_reply, "
                    "clarification, enthusiastic_accept, soft_decline, neutral"
                ),
            },
            {
                "role": "user",
                "content": f"Message: {message!r}. Language context: {languages}",
            },
        ],
        max_tokens=64,
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    intent = str(data.get("intent") or "neutral").strip().lower()
    allowed = {
        "accept",
        "decline",
        "join_intent",
        "question",
        "auto_reply",
        "clarification",
        "enthusiastic_accept",
        "soft_decline",
        "neutral",
    }
    return intent if intent in allowed else "neutral"


def _heuristic_intent(message: str) -> str:
    text = (message or "").strip().lower()
    if is_auto_reply(text):
        return "auto_reply"
    if re.search(r"\b(join|start|onboard|sign me|register|setup|set up|activate)\b", text):
        return "join_intent"
    if re.search(r"\b(yes|yep|sure|go|do it|ok|okay|haan|ha|karo|kar do|proceed|please do)\b", text):
        if re.search(r"\b(great|awesome|perfect|love|jaldi|abhi)\b", text):
            return "enthusiastic_accept"
        return "accept"
    if re.search(r"\b(no|nope|stop|unsubscribe|not interested|mat|nahi|nahin|band)\b", text):
        return "decline"
    if re.search(r"\b(later|baad|kal|not now|abhi nahi|busy)\b", text):
        return "soft_decline"
    if "?" in text or re.search(r"\b(what|why|how|when|cost|price|kya|kaise|kitna|kab)\b", text):
        return "question"
    if re.search(r"\b(which|mean|explain|clarify|samjhao|detail)\b", text):
        return "clarification"
    return "neutral"


def classify_intent(message: str, languages: str) -> str:
    if is_auto_reply(message):
        return "auto_reply"
    if not has_llm_credentials():
        return _heuristic_intent(message)
    try:
        return _classify_with_llm(message, languages)
    except Exception:
        return _heuristic_intent(message)


def _merchant_name(merchant: dict[str, Any] | None) -> str:
    return str(
        _deep_get(
            merchant,
            "owner_name",
            "merchant_name",
            "business_name",
            "name",
            "identity.owner_name",
            "identity.business_name",
        )
        or "there"
    )


def _business_name(merchant: dict[str, Any] | None) -> str:
    return str(
        _deep_get(merchant, "business_name", "identity.business_name", "name", "merchant_name")
        or "your business"
    )


def _category_from_store(
    store: ContextStore, merchant: dict[str, Any] | None, trigger: dict[str, Any] | None
) -> dict[str, Any]:
    candidates = [
        _deep_get(trigger, "category_id", "category_slug", "category"),
        _deep_get(merchant, "category_id", "category_slug", "business_category", "category"),
        _deep_get(merchant, "identity.category_id", "identity.category_slug"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        found = store.get("category", str(candidate))
        if found:
            return found
        wanted = str(candidate).lower()
        for _, payload, _ in store.list_contexts("category"):
            slug = str(
                payload.get("slug")
                or payload.get("category_slug")
                or payload.get("id")
                or payload.get("name")
                or ""
            ).lower()
            if slug == wanted:
                return payload
    contexts = store.list_contexts("category")
    return contexts[0][1] if contexts else {}


def _get_trigger(store: ContextStore, conv: dict[str, Any] | None) -> dict[str, Any]:
    trigger_id = (conv or {}).get("trigger_id") or (conv or {}).get("metadata", {}).get("trigger_id")
    if not trigger_id:
        return {}
    return store.get("trigger", str(trigger_id)) or {}


def _last_assistant_body(history: list[dict[str, Any]]) -> str:
    for entry in reversed(history):
        if entry.get("role") in {"assistant", "vera"}:
            return str(entry.get("body") or entry.get("message") or "")
    return ""


def _last_user_intents(history: list[dict[str, Any]], limit: int = 3) -> list[str]:
    intents: list[str] = []
    for entry in reversed(history):
        if entry.get("role") in {"merchant", "customer", "user"} and entry.get("intent"):
            intents.append(str(entry["intent"]))
            if len(intents) >= limit:
                break
    return intents


def _append_assistant(store: ContextStore, conv_id: str, body: str, cta: str, rationale: str) -> None:
    store.append_history(
        conv_id,
        {
            "role": "assistant",
            "at": _utc_now(),
            "body": body,
            "cta": cta,
            "rationale": rationale,
        },
    )


def _shorten(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _hinglish_suffix(language: str, body: str) -> str:
    if language in {"hi", "hi-en"} and not HINDI_WORD_RE.search(body):
        return body.rstrip(".") + " - theek hai?"
    return body


def _answer_from_context(
    message: str,
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    category: dict[str, Any],
) -> str:
    offer = _first_offer(merchant, category) or "the current offer"
    facts = extract_key_facts(category, merchant, trigger)
    fact = facts[0] if facts else f"{offer} is active"
    text = message.lower()
    if "price" in text or "cost" in text or "kitna" in text or "₹" in text:
        return f"{offer}. I can turn this into a customer reply template if you say YES."
    if "why" in text or "kyu" in text or "kya" in text:
        return f"Because this trigger is live now: {fact}. Want me to handle the draft?"
    return f"Based on your context: {fact}. I can draft the next message from this - reply YES."


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def _reply_with_llm(
    intent: str,
    message: str,
    language: str,
    history: list[dict[str, Any]],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    category: dict[str, Any],
    customer: dict[str, Any] | None,
) -> dict[str, Any]:
    client = get_llm_client()
    route = route_for_trigger(trigger)
    system_prompt = f"""
You are Vera continuing a WhatsApp conversation with an Indian merchant.
Intent: {intent}
Route: {route}
Language to match: {language}

Rules:
- Return JSON only with body, cta, rationale.
- body <= 320 chars.
- Do not re-introduce Vera after turn 1.
- Do not repeat the opening hook.
- If merchant said YES to X, do X in this message and report the result.
- Answer questions directly from merchant/category/trigger context.
- Use one CTA only.
- No URLs and no fabricated numbers.
""".strip()
    user_prompt = f"""
LAST 4 TURNS:
{_json_compact(history[-4:], 2500)}

MERCHANT CONTEXT:
{_json_compact(merchant, 3000)}

CATEGORY CONTEXT:
{_json_compact(category, 2500)}

TRIGGER CONTEXT:
{_json_compact(trigger, 2500)}

CUSTOMER CONTEXT:
{_json_compact(customer, 1500)}

MERCHANT'S LATEST MESSAGE:
{message}
""".strip()
    response = client.chat.completions.create(
        model=REPLY_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=384,
    )
    return json.loads(response.choices[0].message.content or "{}")


def _fallback_reply(
    intent: str,
    message: str,
    language: str,
    history: list[dict[str, Any]],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    category: dict[str, Any],
) -> dict[str, str]:
    name = _merchant_name(merchant)
    business = _business_name(merchant)
    offer = _first_offer(merchant, category) or "your current offer"
    facts = extract_key_facts(category, merchant, trigger)
    fact = facts[0] if facts else "your latest profile signal"
    last_hook = _last_assistant_body(history)

    if intent in {"accept", "enthusiastic_accept"}:
        body = (
            f"Done, {name}. Draft ready for {business}: '{offer} is live today - "
            "message us to book.' Want me to make it a Google post too?"
        )
        cta = "Reply POST"
        rationale = "Accepted request; reported a completed draft instead of re-selling."
    elif intent == "join_intent":
        body = (
            f"Great, {name}. You're in. Send me one offer or service for this week "
            "and I'll build your first Google post + WhatsApp reply."
        )
        cta = "Send one offer"
        rationale = "Join intent goes straight to onboarding."
    elif intent in {"question", "clarification"}:
        body = _answer_from_context(message, merchant, trigger, category)
        cta = "Reply YES"
        rationale = "Answered from stored context and offered one next step."
    elif intent == "neutral":
        if last_hook:
            body = f"Quick one, {name}: {fact}. I can handle the draft around {offer}; just say GO."
        else:
            body = f"{name}, {fact}. Want me to draft the next message around {offer}?"
        cta = "Say GO"
        rationale = "Neutral reply received; sent a smaller next-best nudge."
    else:
        body = f"Noted, {name}. I won't push this now."
        cta = ""
        rationale = "Fallback for non-send intent."

    body = _hinglish_suffix(language, body)
    return {"body": _shorten(body), "cta": cta, "rationale": rationale}


def compose_reply(
    intent: str,
    message: str,
    language: str,
    history: list[dict[str, Any]],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    category: dict[str, Any],
    customer: dict[str, Any] | None,
) -> dict[str, str]:
    if has_llm_credentials():
        try:
            result = _reply_with_llm(
                intent, message, language, history, merchant, trigger, category, customer
            )
            body = _shorten(str(result.get("body") or ""))
            if body:
                return {
                    "body": body,
                    "cta": str(result.get("cta") or "Reply YES"),
                    "rationale": str(result.get("rationale") or "LLM reply from context."),
                }
        except Exception:
            pass
    return _fallback_reply(intent, message, language, history, merchant, trigger, category)


def handle_reply(
    store: ContextStore,
    conversation_id: str,
    merchant_id: str,
    customer_id: str | None,
    from_role: str,
    message: str,
    received_at: str | None,
    turn_number: int,
) -> dict[str, Any]:
    conv = store.get_conversation(conversation_id)
    if conv is None:
        store.create_conversation(
            conversation_id,
            merchant_id,
            None,
            customer_id=customer_id,
            metadata={
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "turn": turn_number,
                "created_from": "reply",
            },
        )
        conv = store.get_conversation(conversation_id)

    history = list((conv or {}).get("history") or [])
    effective_customer_id = (
        customer_id
        or (conv or {}).get("customer_id")
        or (conv or {}).get("metadata", {}).get("customer_id")
    )
    merchant = store.get("merchant", merchant_id) or {}
    trigger = _get_trigger(store, conv)
    category = _category_from_store(store, merchant, trigger)
    customer = store.get("customer", effective_customer_id) if effective_customer_id else None
    language = detect_reply_language(message)
    intent = classify_intent(message, language)

    incoming = {
        "role": from_role or "merchant",
        "at": received_at or _utc_now(),
        "message": message,
        "intent": intent,
        "language": language,
        "turn_number": turn_number,
    }
    store.append_history(
        conversation_id,
        incoming,
        merchant_id=merchant_id,
        customer_id=effective_customer_id,
        trigger_id=(conv or {}).get("trigger_id"),
    )
    history.append(incoming)

    if intent == "auto_reply":
        if turn_number <= 2:
            name = _merchant_name(merchant)
            body = _shorten(
                f"Hi, can the owner/manager see this? {name}, I have one 5-min Google profile task ready from today's signal. Reply YES and I'll do it."
            )
            cta = "Reply YES"
            rationale = "Detected WhatsApp Business auto-reply; one human re-engagement attempt allowed."
            _append_assistant(store, conversation_id, body, cta, rationale)
            return {"action": "send", "body": body, "cta": cta, "rationale": rationale}
        return {
            "action": "end",
            "rationale": "Detected repeated auto-reply at turn 3+; ending gracefully.",
        }

    if intent in {"decline"}:
        return {
            "action": "end",
            "rationale": "Merchant declined or opted out; ending without another prompt.",
        }

    if intent in {"soft_decline"}:
        return {
            "action": "wait",
            "wait_seconds": 24 * 60 * 60,
            "rationale": "Merchant deferred; waiting 24 hours before any follow-up.",
        }

    if turn_number >= 5 and intent not in {"accept", "enthusiastic_accept", "join_intent"}:
        return {
            "action": "end",
            "rationale": "Turn limit reached without accept or join intent.",
        }

    recent_intents = _last_user_intents(history, 3)
    if turn_number >= 3 and len(recent_intents) >= 3 and all(
        intent == "neutral" for intent in recent_intents[:3]
    ):
        return {
            "action": "wait",
            "wait_seconds": 30 * 60,
            "rationale": "Three consecutive neutral turns; backing off for 30 minutes.",
        }

    reply = compose_reply(
        intent, message, language, history, merchant, trigger, category, customer
    )
    body = reply["body"]
    cta = reply.get("cta") or "Reply YES"
    rationale = reply.get("rationale") or f"Handled intent={intent}."
    _append_assistant(store, conversation_id, body, cta, rationale)
    return {"action": "send", "body": body, "cta": cta, "rationale": rationale}
