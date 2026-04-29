# Vera Merchant AI Assistant

FastAPI implementation of Vera for the magicpin AI Challenge. The service stores category, merchant, trigger, and customer context in SQLite, composes WhatsApp-ready actions from `/v1/tick`, and continues conversations through `/v1/reply`.

## Approach

Vera combines four layers of context:

- `CategoryContext` for tone, taboo words, peer benchmarks, digest items, and offer patterns.
- `MerchantContext` for identity, metrics, active offers, and history.
- `TriggerContext` for why the message is going now.
- `CustomerContext` for customer-facing recall flows.

`composer.py` maps trigger kinds into route-specific prompts, extracts a `KEY FACTS` block before generation, includes scored few-shots per route, calls Azure OpenAI with JSON output, and validates hard constraints such as body length, no URLs, taboo vocabulary, and valid `send_as`.

`multi_turn.py` detects WhatsApp Business auto-replies, classifies merchant intent with `gpt-4.1-mini`, enforces turn limits, and replies from the stored conversation plus merchant context. If no OpenAI or Azure OpenAI API key is configured during local testing, deterministic fallbacks still return valid JSON.

## Model

- Composition: Azure OpenAI deployment `gpt-4.1`, temperature `0`, JSON response format.
- Intent classification: Azure OpenAI deployment `gpt-4.1-mini`, temperature `0`, JSON response format.
- Retries: `tenacity` exponential backoff, up to 3 attempts.

## Azure OpenAI

The app uses the official OpenAI Python SDK's Azure client when `AZURE_OPENAI_API_KEY` is set.

```env
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
# Azure uses deployment names here, not necessarily the base model names.
AZURE_OPENAI_COMPOSE_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_CLASSIFY_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_REPLY_DEPLOYMENT=gpt-4.1
OPENAI_REPLY_MODEL=gpt-4.1
OPENAI_TIMEOUT_SECONDS=8
SUPPRESSION_TTL_SECONDS=86400
```

For Azure OpenAI, the `model` parameter is the deployment name. If your Azure portal deployment is named `my-gpt4`, set `AZURE_OPENAI_COMPOSE_DEPLOYMENT=my-gpt4` even if the underlying model is GPT-4.1.

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Endpoints

- `POST /v1/context` stores versioned context. Same or lower version returns `409`.
- `POST /v1/tick` inspects available triggers and returns up to 5 send actions.
- `POST /v1/reply` handles merchant/customer replies and returns `send`, `wait`, or `end`.
- `GET /v1/healthz` returns uptime and context counts.
- `GET /v1/metadata` returns submission metadata.

## Persistence

SQLite tables:

- `contexts(key, version, payload)`
- `suppressions(key, sent_at)`
- `conversations(conv_id, merchant_id, customer_id, trigger_id, history)`

Set `VERA_DB_PATH` to change the database file. `SUPPRESSION_TTL_SECONDS` controls duplicate-send memory, defaulting to 24 hours. On startup the app attempts to preload JSON contexts from `./expanded/` at version `0`, if that directory exists.

## Tradeoffs

The prompt carries compact JSON context and a fact block instead of a fully normalized schema, because challenge fixtures may vary. The deterministic fallback keeps the API robust for smoke tests, but best message quality requires a valid OpenAI key. Suppression keys are stable across merchant, customer, trigger, route, and trigger date when available.
