import asyncio
import hashlib
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from composer import COMPOSE_MODEL, compose
from multi_turn import CLASSIFY_MODEL, handle_reply
from store import ContextStore
from utils import deep_get


load_dotenv()

STARTED_AT = monotonic()
STORE = ContextStore(os.getenv("VERA_DB_PATH", "vera_context.db"))


class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int = Field(ge=0)
    payload: dict[str, Any]
    delivered_at: str | None = None


class TickRequest(BaseModel):
    now: str | None = None
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str | None = None
    turn_number: int = Field(default=1, ge=0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ack_id(scope: str, context_id: str, version: int) -> str:
    raw = f"{scope}:{context_id}:{version}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _candidate_id(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = deep_get(payload, key)
        if value not in (None, "", []):
            return str(value)
    return None


def _scope_from_path(path: Path) -> str | None:
    text = " ".join(part.lower() for part in path.parts)
    if "categor" in text:
        return "category"
    if "merchant" in text or "business" in text:
        return "merchant"
    if "customer" in text or "patient" in text:
        return "customer"
    if "trigger" in text or "event" in text:
        return "trigger"
    return None


def _scope_id_for_record(record: dict[str, Any], path: Path) -> tuple[str | None, str | None, int, dict[str, Any]]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
    scope = str(record.get("scope") or payload.get("scope") or _scope_from_path(path) or "").lower()
    if scope in {"categories", "category_contexts"}:
        scope = "category"
    elif scope in {"merchants", "merchant_contexts"}:
        scope = "merchant"
    elif scope in {"customers", "customer_contexts"}:
        scope = "customer"
    elif scope in {"triggers", "trigger_contexts"}:
        scope = "trigger"
    elif scope.endswith("s"):
        scope = scope[:-1]
    context_id = (
        record.get("context_id")
        or payload.get("context_id")
        or payload.get("id")
        or payload.get(f"{scope}_id")
        or payload.get(f"{scope}_slug")
        or payload.get("slug")
        or payload.get("merchant_id")
        or payload.get("customer_id")
        or payload.get("trigger_id")
    )
    version = int(record.get("version") or payload.get("version") or 0)
    return (scope or None), (str(context_id) if context_id else None), version, payload


def _iter_dataset_records(data: Any, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return records

    if any(key in data for key in ("scope", "context_id", "payload", "merchant_id", "trigger_id", "customer_id")):
        return [data]

    scope_keys = {
        "categories": "category",
        "category": "category",
        "merchants": "merchant",
        "merchant": "merchant",
        "customers": "customer",
        "customer": "customer",
        "triggers": "trigger",
        "trigger": "trigger",
    }
    for key, scope in scope_keys.items():
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("scope", scope)
                    records.append(item)
        elif isinstance(value, dict):
            for context_id, payload in value.items():
                if isinstance(payload, dict):
                    item = dict(payload)
                    item.setdefault("scope", scope)
                    item.setdefault("context_id", context_id)
                    records.append(item)

    if records:
        return records

    guessed_scope = _scope_from_path(path)
    if guessed_scope:
        for context_id, payload in data.items():
            if isinstance(payload, dict):
                item = dict(payload)
                item.setdefault("scope", guessed_scope)
                item.setdefault("context_id", context_id)
                records.append(item)
    return records


def preload_expanded_dataset(store: ContextStore, root: Path = Path("expanded")) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    loaded = 0
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("[vera] skipping %s: %s", path, exc)
            continue
        for record in _iter_dataset_records(data, path):
            scope, context_id, version, payload = _scope_id_for_record(record, path)
            if scope not in {"category", "merchant", "customer", "trigger"} or not context_id:
                continue
            accepted, _ = store.upsert(scope, context_id, version, payload)
            if accepted:
                loaded += 1
    return loaded


def _normalize_category_id(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _merchant_matches_category(merchant: dict[str, Any], category_id: str | None) -> bool:
    if not category_id:
        return True
    wanted = _normalize_category_id(category_id)
    candidates = [
        deep_get(merchant, "category_id", "category_slug", "category", "business_category"),
        deep_get(merchant, "identity.category_id", "identity.category_slug", "identity.category"),
    ]
    return any(_normalize_category_id(candidate) == wanted for candidate in candidates if candidate)


def _resolve_category(store: ContextStore, merchant: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        _candidate_id(trigger, "category_id", "category_slug", "category"),
        _candidate_id(merchant, "category_id", "category_slug", "business_category", "category"),
        _candidate_id(merchant, "identity.category_id", "identity.category_slug", "identity.category"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        found = store.get("category", candidate)
        if found:
            return found
        wanted = _normalize_category_id(candidate)
        for _, payload, _ in store.list_contexts("category"):
            keys = [
                payload.get("id"),
                payload.get("slug"),
                payload.get("category_id"),
                payload.get("category_slug"),
                payload.get("name"),
            ]
            if any(_normalize_category_id(key) == wanted for key in keys if key):
                return payload
    categories = store.list_contexts("category")
    return categories[0][1] if categories else {}


def _resolve_customer(store: ContextStore, trigger: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    customer_id = _candidate_id(trigger, "customer_id", "customer.context_id", "customer.id")
    if customer_id:
        return customer_id, store.get("customer", customer_id) or trigger.get("customer")
    customer = trigger.get("customer")
    if isinstance(customer, dict):
        inferred = _candidate_id(customer, "customer_id", "id", "context_id")
        return inferred, customer
    return None, None


def _trigger_id(trigger: dict[str, Any], fallback: str) -> str:
    return str(trigger.get("trigger_id") or trigger.get("id") or fallback)


def _pre_suppression_key(trigger: dict[str, Any], merchant_id: str | None, customer_id: str | None, fallback_trigger_id: str) -> str:
    if trigger.get("suppression_key"):
        return str(trigger["suppression_key"])
    kind = trigger.get("kind") or trigger.get("type") or "generic"
    return f"pre:{merchant_id or 'merchant'}:{customer_id or 'merchant'}:{fallback_trigger_id}:{kind}"


def _candidate_merchants_for_trigger(store: ContextStore, trigger: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    merchant_id = _candidate_id(trigger, "merchant_id", "merchant.context_id", "merchant.id")
    if merchant_id:
        merchant = store.get("merchant", merchant_id) or trigger.get("merchant")
        return [(merchant_id, merchant)] if isinstance(merchant, dict) else []

    merchant = trigger.get("merchant")
    if isinstance(merchant, dict):
        inferred = _candidate_id(merchant, "merchant_id", "id", "context_id") or f"merchant-{uuid.uuid4().hex[:8]}"
        return [(inferred, merchant)]

    category_id = _candidate_id(trigger, "category_id", "category_slug", "category")
    merchants = []
    seen_merchant_ids: set[str] = set()
    for context_id, payload, _ in store.list_contexts("merchant"):
        payload_id = _candidate_id(payload, "merchant_id", "id", "identity.merchant_id") or context_id
        normalized_id = str(payload_id).strip().lower()
        if normalized_id in seen_merchant_ids:
            continue
        if _merchant_matches_category(payload, category_id):
            seen_merchant_ids.add(normalized_id)
            merchants.append((context_id, payload))
        if len(merchants) >= 5:
            break
    return merchants


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, preload_expanded_dataset, STORE)
    yield
    STORE.close()


app = FastAPI(title="Vera Merchant AI Assistant", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def internal_exception_handler(_, exc: Exception):
    detail = str(exc) if os.getenv("VERA_DEBUG_ERRORS") == "1" else "internal_server_error"
    return JSONResponse(
        status_code=500,
        content={"error": "internal", "detail": detail},
    )


def _upsert_sync(request: ContextRequest):
    accepted, current_version = STORE.upsert(
        request.scope, request.context_id, request.version, request.payload
    )
    if not accepted:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )
    return {
        "accepted": True,
        "ack_id": ack_id(request.scope, request.context_id, request.version),
        "stored_at": utc_now(),
    }


@app.post("/v1/context")
async def upsert_context(request: ContextRequest):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _upsert_sync, request)


def _build_tick_action(
    request: TickRequest,
    candidate: dict[str, Any],
    composed: dict[str, Any],
) -> dict[str, Any]:
    merchant_id = candidate["merchant_id"]
    customer_id = candidate["customer_id"]
    trigger_id = candidate["trigger_id"]
    category = candidate["category"]
    body = composed["body"]
    cta = composed["cta"]
    send_as = composed["send_as"]
    route = composed.get("route") or "generic"
    suppression_key = composed["suppression_key"]
    conversation_id = f"conv_{uuid.uuid4().hex}"
    STORE.create_conversation(
        conversation_id,
        merchant_id,
        trigger_id,
        customer_id=customer_id,
        metadata={
            "merchant_id": merchant_id,
            "trigger_id": trigger_id,
            "category_slug": (
                category.get("slug")
                or category.get("category_slug")
                or category.get("id")
            ),
            "customer_id": customer_id,
            "turn": 0,
            "suppression_key": suppression_key,
        },
        first_message={
            "role": "assistant",
            "at": request.now or utc_now(),
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "route": route,
        },
    )
    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": send_as,
        "trigger_id": trigger_id,
        "template_name": route,
        "template_params": [route]
        + [str(fact) for fact in composed.get("key_facts", [])[:3]],
        "body": body,
        "cta": cta,
        "suppression_key": suppression_key,
        "rationale": composed.get("rationale", ""),
    }


def _tick_sync(request: TickRequest) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for available_id in request.available_triggers:
        if len(candidates) >= 5:
            break
        trigger = STORE.get("trigger", available_id)
        if trigger is None:
            continue
        trigger_id = _trigger_id(trigger, available_id)
        customer_id, customer = _resolve_customer(STORE, trigger)

        for merchant_id, merchant in _candidate_merchants_for_trigger(STORE, trigger):
            if len(candidates) >= 5:
                break
            if not merchant:
                continue
            pre_key = _pre_suppression_key(trigger, merchant_id, customer_id, trigger_id)
            if not STORE.reserve_suppression(pre_key):
                continue
            category = _resolve_category(STORE, merchant, trigger)
            candidates.append(
                {
                    "merchant_id": merchant_id,
                    "merchant": merchant,
                    "customer_id": customer_id,
                    "customer": customer,
                    "trigger_id": trigger_id,
                    "trigger": trigger,
                    "category": category,
                    "pre_key": pre_key,
                }
            )
    if not candidates:
        return []

    actions: list[dict[str, Any]] = []
    composed_by_index: dict[int, dict[str, Any]] = {}
    max_workers = min(5, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                compose,
                candidate["category"],
                candidate["merchant"],
                candidate["trigger"],
                candidate["customer"],
            ): index
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                composed_by_index[index] = future.result()
            except Exception:
                continue

    for index, candidate in enumerate(candidates):
        if len(actions) >= 5:
            break
        composed = composed_by_index.get(index)
        if not composed:
            STORE.clear_suppression(candidate["pre_key"])
            continue
        suppression_key = composed.get("suppression_key")
        if not suppression_key:
            STORE.clear_suppression(candidate["pre_key"])
            continue
        if suppression_key != candidate["pre_key"] and not STORE.reserve_suppression(suppression_key):
            STORE.clear_suppression(candidate["pre_key"])
            continue
        try:
            actions.append(_build_tick_action(request, candidate, composed))
        except Exception:
            STORE.clear_suppression(suppression_key)
            STORE.clear_suppression(candidate["pre_key"])
            continue
    return actions


@app.post("/v1/tick")
async def tick(request: TickRequest):
    loop = asyncio.get_running_loop()
    actions = await loop.run_in_executor(None, _tick_sync, request)
    return {"actions": actions}


def _reply_sync(request: ReplyRequest):
    return handle_reply(
        STORE,
        conversation_id=request.conversation_id,
        merchant_id=request.merchant_id,
        customer_id=request.customer_id,
        from_role=request.from_role,
        message=request.message,
        received_at=request.received_at,
        turn_number=request.turn_number,
    )


@app.post("/v1/reply")
async def reply(request: ReplyRequest):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _reply_sync, request)


@app.get("/v1/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(monotonic() - STARTED_AT),
        "contexts_loaded": STORE.counts(),
    }


@app.get("/v1/metadata")
def metadata():
    members = os.getenv("TEAM_MEMBERS", "")
    return {
        "team_name": os.getenv("TEAM_NAME", "Vera Codex"),
        "team_members": [m.strip() for m in members.split(",") if m.strip()],
        "model": f"{COMPOSE_MODEL} for composition, {CLASSIFY_MODEL} for intent classification",
        "approach": (
            "Route triggers into specialized prompt templates, extract key facts "
            "before generation, validate hard WhatsApp constraints, persist "
            "contexts/suppressions/conversations in SQLite, and use multi-turn "
            "intent handling for replies. Azure OpenAI is used when "
            "AZURE_OPENAI_API_KEY is configured."
        ),
        "contact_email": os.getenv("CONTACT_EMAIL", "team@example.com"),
        "version": "1.0.0",
        "submitted_at": os.getenv("SUBMITTED_AT", "2026-04-29T00:00:00+05:30"),
    }
