# Polymarket AI Trading Bot

AI research workspace for Polymarket prediction markets. The trading bot decides with a machine-learning ensemble and an OpenAI language model, then records a paper bet. A second binary copies leaderboard wallets with fixed rules and no model.

Two Rust processes share a library, a Postgres ledger, and a Python model server. They read public Polymarket data. Bets are ledger rows. The CLOB is used for prices and books.

- **trading-bot (AI).** An XGBoost stacking ensemble (XGBoost, LightGBM, HistGradientBoosting, ExtraTrees, RandomForest, and a meta-learner) scores each market. A Bayesian step anchors that probability to the market price. OpenAI (`gpt-4o` by default) checks open bets for correlation and can assess news. Fractional Kelly sizes the paper stake. `OPENAI_API_KEY` is required for this process to start.
- **copy-trading-bot.** Watches the public leaderboard and records a paper copy when a followed trader opens a new position. No ensemble and no LLM.

Each process has its own Telegram bot. They share one database.

<p align="center">
  <img src="./assets/ai-trading-bot-flow.png" width="100%" alt="Flow of the AI trading bot: Gamma markets, 29 features, the Python ensemble, OpenAI, entry gates, Kelly sizing, a paper Postgres bet, and actions the bot does not take" />
</p>

<p align="center">
  <img src="./assets/copy-trading-bot-flow.png" width="100%" alt="Flow of the copy-trading bot: leaderboard source, follow commands, new buy detection, staleness filter, Kelly sizing, a paper copy, Telegram, and actions the bot does not take" />
</p>

This is experimental software for learning. The ledger can lose simulated money. It is not a production trading system.

## Architecture

```mermaid
graph TB
    subgraph Workspace["Cargo workspace"]
        Common["polymarket-common"]
        TB["trading-bot"]
        CTB["copy-trading-bot"]
        TB --> Common
        CTB --> Common
    end

    subgraph External["External services"]
        Gamma["Gamma API"]
        CLOB["CLOB API and websocket"]
        News["News RSS"]
        OpenAI["OpenAI API"]
        Data["Data API"]
    end

    subgraph Infra["Local infrastructure"]
        PG[(PostgreSQL)]
        Sidecar["Model server"]
        Prom["Prometheus"]
        Graf["Grafana"]
    end

    TB --> Gamma & CLOB & News & OpenAI
    TB --> PG & Sidecar
    CTB --> Gamma & CLOB & Data
    CTB --> PG
    TB -.->|metrics :9000| Prom
    CTB -.->|metrics :9001| Prom
    Prom --> Graf
```

`trading-bot` needs `OPENAI_API_KEY` at startup. The scanner builds an OpenAI client before the loops start. Model scores come from the Python sidecar. An empty `MODEL_SIDECAR_URL` disables them: the batch predictor returns no probabilities and the funnel records no model bets. A URL that fails its health check is still used; later predict calls fail until the server answers. The live path does not fall back to `model/xgb.rs`. That module is a JSON tree walker for tests and offline scoring.

## Repository layout

```
polymarket-bot/
├── common/                 # polymarket-common
│   └── src/
│       ├── data/           # Gamma models and historical crawler
│       ├── model/          # MarketFeatures v5 (29 features)
│       ├── pricing/        # Fractional Kelly
│       ├── storage/        # Postgres ledger and copy-trade queries
│       ├── telegram/       # Send, broadcast, getUpdates
│       ├── signal.rs
│       ├── format.rs
│       └── metrics.rs
├── trading-bot/src/
│   ├── main.rs             # live (default), test, backtest
│   ├── config.rs
│   ├── strategy.rs         # aggressive, balanced, conservative
│   ├── bayesian.rs
│   ├── calibration.rs
│   ├── scanner/            # Gamma scan, websocket, news
│   ├── cycles/             # bet scan, housekeeping, alerts, heartbeat
│   ├── model/              # Rust XGBoost JSON trees and sidecar client
│   ├── backtest/
│   └── telegram/commands.rs
├── copy-trading-bot/src/
│   ├── scanner/copy_trader.rs
│   ├── cycles/             # copy, housekeeping, daily digest
│   └── telegram/commands.rs
├── scripts/                # fetch_data.py, train_model.py, serve_model.py
├── migrations/
├── monitoring/
└── adr/
```

## Trading bot

### Scan

```mermaid
flowchart LR
    A[Gamma markets] --> B[Volume, expiry, price filters]
    B --> C[29 features]
    C --> D[Ensemble or local XGBoost]
    D --> E[Bayesian anchor]
    E --> F[Sports and YES filters]
    F --> G[LLM correlation check]
    G --> H[Per-strategy thresholds]
    H --> I[Kelly size]
    I --> J[Paper bet and Telegram]
```

1. **Fetch.** Gamma returns active markets. Defaults keep volume at or above $1,000, expiry inside 14 days, and YES price between 0.03 and 0.97. Up to 1,000 markets are fetched per scan.
2. **Features.** Each candidate becomes the v5 vector in `common/src/model/features.rs`. Order matches `scripts/train_model.py`.
3. **Score.** With `MODEL_SIDECAR_URL` set, each vector goes to the sidecar. The ensemble is XGBoost, LightGBM, HistGradientBoosting, ExtraTrees, RandomForest, and a meta-learner. Markets with no probability are skipped. Without the URL, this step produces no scores.
4. **Anchor.** The model probability is pulled toward the market price. The likelihood ratio is damped as `LR^(confidence × LR_DAMPING)` with `LR_DAMPING` defaulting to 0.5.
5. **Filters.** Sports and esports markets are dropped when `BLOCK_SPORTS=true`. YES-side XGBoost bets are dropped when `BLOCK_YES_SIDE=true`. Signals whose Kelly fraction is below `MIN_KELLY_SIZE` (0.02) or whose entry price is below `MIN_BET_PRICE` (0.15) are dropped.
6. **Correlation.** An LLM pass compares the candidate with open paper positions and looks for duplicate or mutually exclusive exposure. A failed call fails open: the bet is still eligible.
7. **Strategies.** Aggressive, balanced, and conservative each apply their own edge, confidence, daily signal cap, and minimum stake. They do not share a bankroll.
8. **Size.** Stake is fractional Kelly, scaled down as expiry approaches, then reduced by the configured slippage (1%) and fee (2%). Those percentages are ledger assumptions. They are not transfers.
9. **Record.** `place_bet` inserts a row and deducts `cost + fee` from that strategy’s bankroll and from the global `bankroll` key. A feature snapshot is stored for later retraining. Telegram receives the stake, edge, confidence, and reasoning.

### Loops

| Loop | Default | Work |
|------|---------|------|
| Bet scan | 10 min | Rank, filter, size, record |
| News | 10 min | RSS plus OpenAI embeddings, only when `NEWS_ENABLED=true` |
| Housekeeping | 30 min | Resolve settled markets, optional stop-loss and expiry exits, calibration, Brier log, daily report |
| Heartbeat | 60 min | Open bets, scan counts, per-strategy results. `0` disables it |
| Websocket | continuous | Re-score a market after a move of about 3% or more. Open-bet price alerts use about 5% and a one-hour cooldown |

`cargo run -p trading-bot -- test` forces the scan and news intervals to 2 minutes. `cargo run -p trading-bot -- backtest` runs the historical engine instead of the live loops.

News is off by default. With a sidecar, news does not change the ensemble score. Leave it off unless you want the RSS and embedding cost.

### Strategies

Three profiles run together. Each starts at `STRATEGY_BANKROLL` (300). The unit in configuration comments is EUR; the ledger stores a number, not a currency conversion.

| Strategy | Kelly fraction | Min edge | Min confidence | Max signals / day | Min bet |
|----------|----------------|----------|----------------|-------------------|---------|
| Aggressive | 0.50 | 0.05 | 0.40 | 10 | 5 |
| Balanced | 0.25 | 0.06 | 0.40 | 5 | 5 |
| Conservative | 0.15 | 0.08 | 0.50 | 3 | 15 |

Override any cell with `AGGRESSIVE_*`, `BALANCED_*`, or `CONSERVATIVE_*`. `STRATEGIES` selects which names are active.

Stop-loss and expiry exits are off. `STOP_LOSS_PCT=999` means “do not exit on unrealized loss.” `EXIT_DAYS_BEFORE_EXPIRY=0` means “do not exit early near expiry.” Housekeeping still marks bets resolved when Gamma reports an outcome. A resolved loss books `-(cost + entry fee)`. A win pays the share payout minus a 2% exit fee, then subtracts cost and the entry fee.

### Telegram

| Command | Response |
|---------|----------|
| `/start` | Short command list |
| `/stats` | Portfolio totals and per-strategy breakdown |
| `/open` | Open paper positions, mark, PnL, Polymarket links |
| `/brier` | Model Brier score against the market-price baseline |
| `/health` | Uptime, scan counts, signals, sidecar age |
| `/help` | Command list |

Outbound messages cover new bets, resolutions (a win can include a Giphy animation), websocket bets, the hourly heartbeat, and the daily report. `TELEGRAM_CHAT_ID` is the owner chat. Subscribers who talk to the bot are stored in `telegram_users` and can receive broadcasts.

## Copy-trading bot

```mermaid
flowchart LR
    A[Leaderboard and activity] --> B[New buys from followed wallets]
    B --> C[Staleness and price drift]
    C --> D[Kelly size]
    D --> E[Paper bet with CopyRef]
    E --> F[Telegram]
```

The process polls the Polymarket data API about once a minute (`COPY_TRADE_INTERVAL_MINS`, default 1). A followed wallet’s new BUY is copied only after the staleness and price-drift checks in `scanner/copy_trader.rs`. The paper row stores a `CopyRef` back to the source trader. The same 1% slippage and 2% fee assumptions apply.

`/follow` and `/unfollow` are limited to the owner chat.

| Command | Response |
|---------|----------|
| `/stats` | Copy-trading results |
| `/copy` | Open copy positions |
| `/traders` | Followed wallets |
| `/leaderboard` | Top traders for day, month, and all-time |
| `/follow <wallet>` | Follow a trader |
| `/unfollow <wallet>` | Stop following a trader |
| `/subscribers` | Subscriber list |
| `/help` | Command list |

A daily digest of the top traders is on by default: 5 names, 09:00 UTC. `DIGEST_ENABLED`, `DIGEST_TOP_N`, and `DIGEST_HOUR_UTC` change that. Use a second Telegram bot token so this process does not share updates with `trading-bot`. Compose maps `COPY_TELEGRAM_BOT_TOKEN` and `COPY_TELEGRAM_CHAT_ID` onto this service.

`COPY_TRADE_ENABLED` appears in `.env.example` and is not read by either binary.

## Model server

`scripts/serve_model.py` is a FastAPI process.

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | Process is up and a model is loaded |
| POST | `/predict` | One feature vector |
| POST | `/predict_batch` | A batch of vectors |
| POST | `/reload` | Load `ensemble.joblib` from disk |
| POST | `/retrain` | Fetch and train. `force=true` ignores the age check |
| GET | `/retrain/status` | Last retrain state |
| POST | `/retrain/warmstart` | Shorter retrain that still writes `ensemble.joblib` |

Artifacts live in `MODEL_DIR` (`/model` in Compose). Training pins `scikit-learn==1.8.0` because the joblib file is pickled with that version. A scheduled retrain runs when the model is older than `RETRAIN_MAX_AGE_HOURS` (24). It pulls `RETRAIN_MARKETS` resolved markets (3000) and, when `DATABASE_URL` is set, folds in this bot’s resolved bets. Warm-start uses `WARMSTART_MARKETS` (500) and is skipped when there is no database. The sidecar loads joblib with `joblib.load`. Treat files in the model volume as code.

### Feature vector (v5)

| Index | Names | Source |
|-------|--------|--------|
| 1–5 | `yes_price`, `momentum_1h`, `momentum_24h`, `volatility_24h`, `rsi` | CLOB price history |
| 6–7 | `log_volume`, `days_to_expiry` | Gamma |
| 8 | `is_crypto` | Keyword category |
| 9–10 | `price_change_1d`, `price_change_1w` | Gamma |
| 11–12 | `days_since_created`, `created_to_expiry_span` | Market timestamps |
| 13 | `is_sports` | Keyword category |
| 14–29 | Question NLP: length, word count, average word length, diversity, number, year, percent, dollar, date, “will”, “by”, “before”, “above”, positive sentiment, negative sentiment, certainty | Question text |

Every prediction can be written to `prediction_log` with the market price, the model posterior, confidence, and edge. After resolution, housekeeping computes a Brier score per source. Skill is `1 - (model_brier / market_brier)`. A positive skill means the model beat the market price on that sample.

## Setup

```bash
cp .env.example .env
```

Required for the trading bot:

```bash
DATABASE_URL=postgres://bot:bot@localhost:5432/polymarket
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
OPENAI_API_KEY=
```

### Docker Compose

```bash
docker compose up -d
```

| Service | What starts |
|---------|-------------|
| `postgres` | Postgres 17. User `bot`, password `bot`, database `polymarket` |
| `model-server` | Image built from `scripts/`. Listens on port 8000 inside the Compose network |
| `bot` | `ghcr.io/skharchikov/polymarket-bot:latest` |
| `copy-trading-bot` | `ghcr.io/skharchikov/polymarket-bot-copy:latest` |
| `prometheus` | Published on port 9090 |
| `grafana` | Published on port 3000. Login `admin` / `admin`. Sign-up is disabled |

The Rust services use the published GHCR images. Editing this checkout does not change those containers. Build and tag your own images, then point Compose at them, when you want the binary to match the source. Metrics ports 9000 and 9001 are bound to `127.0.0.1`. Grafana and Prometheus are published on all interfaces.

### Local binaries

```bash
docker run -d \
  -e POSTGRES_DB=polymarket \
  -e POSTGRES_USER=bot \
  -e POSTGRES_PASSWORD=bot \
  -p 5432:5432 \
  postgres:17-alpine

cargo run -p trading-bot
cargo run -p trading-bot -- test
cargo run -p trading-bot -- backtest
cargo run -p copy-trading-bot
```

Rust edition is 2024. The workspace members are `common`, `trading-bot`, and `copy-trading-bot` (crate version 0.6.41).

## Configuration

### Trading bot

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | required | Postgres URL |
| `TELEGRAM_BOT_TOKEN` | required | Bot token |
| `TELEGRAM_CHAT_ID` | required | Owner chat id |
| `OPENAI_API_KEY` | required to start the scanner | Embeddings, consensus, correlation |
| `MODEL_SIDECAR_URL` | empty | Ensemble base URL. Empty disables model scoring |
| `LLM_MODEL` | `gpt-4o` | Chat model name |
| `STRATEGIES` | `aggressive,balanced,conservative` | Active profiles |
| `STRATEGY_BANKROLL` | `300` | Starting bankroll per strategy |
| `SCAN_INTERVAL_MINS` | `30` | Housekeeping |
| `NEWS_SCAN_INTERVAL_MINS` | `10` | News loop |
| `BET_SCAN_INTERVAL_MINS` | `10` | Scoring loop |
| `NEWS_ENABLED` | `false` | RSS and embeddings |
| `MAX_MARKETS_FETCH` | `1000` | Gamma page cap |
| `MAX_MODEL_CANDIDATES` | `15` | Top model ranks considered |
| `MAX_LLM_CANDIDATES` | `1` | Markets sent to the LLM per cycle |
| `CONSENSUS_AGENTS` | `2` | LLM agents for the fallback path (1–3) |
| `MIN_VOLUME` | `1000` | Minimum market volume |
| `MIN_BOOK_DEPTH` | `200` | Minimum book depth in USD |
| `MIN_PRICE` / `MAX_PRICE` | `0.03` / `0.97` | YES price band |
| `MAX_DAYS_TO_EXPIRY` | `14` | Expiry window |
| `KELLY_FRACTION` | `0.25` | Global Kelly fraction |
| `MIN_EFFECTIVE_EDGE` | `0.08` | Minimum `edge × confidence` |
| `MIN_KELLY_SIZE` | `0.02` | Scanner Kelly floor |
| `MIN_BET_PRICE` | `0.15` | Minimum entry price |
| `SLIPPAGE_PCT` | `0.01` | Paper slippage |
| `FEE_PCT` | `0.02` | Paper fee |
| `LR_DAMPING` | `0.5` | Bayesian damping |
| `BLOCK_SPORTS` | `true` | Drop sports and esports |
| `BLOCK_YES_SIDE` | `true` | Drop YES-side XGBoost bets |
| `CALIBRATION_MIN_SAMPLES` | `20` | Samples before calibration is applied |
| `STOP_LOSS_PCT` | `999` | Unrealized-loss exit. 999 disables it |
| `EXIT_DAYS_BEFORE_EXPIRY` | `0` | Early exit near expiry. 0 disables it |
| `HEARTBEAT_INTERVAL_MINS` | `60` | `0` disables the heartbeat |
| `RETRAIN_INTERVAL_HOURS` | `24` | Age the bot treats as a stale model |
| `METRICS_PORT` | `9000` | Prometheus listener |
| `RUST_LOG` | — | Tracing filter. `.env.example` sets `info` |

### Copy-trading bot

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | required | Postgres URL |
| `TELEGRAM_BOT_TOKEN` | required | Separate bot token |
| `TELEGRAM_CHAT_ID` | required | Owner chat id |
| `COPY_TRADE_INTERVAL_MINS` | `1` | Poll interval |
| `SLIPPAGE_PCT` | `0.01` | Paper slippage |
| `FEE_PCT` | `0.02` | Paper fee |
| `DIGEST_ENABLED` | `true` | Daily top-trader digest |
| `DIGEST_TOP_N` | `5` | Names in the digest |
| `DIGEST_HOUR_UTC` | `9` | Hour the digest is sent |
| `METRICS_PORT` | `9001` | Prometheus listener |

### Model server

Set on the Compose `model-server` service. Defaults are the code defaults.

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | unset | Resolved bets included in training when set |
| `MODEL_DIR` | `/model` | `ensemble.joblib` and training JSON |
| `MODEL_PORT` | `8000` | Listen port |
| `RETRAIN_ON_SCHEDULE` | `true` | Retrain when the file is older than the max age |
| `RETRAIN_MAX_AGE_HOURS` | `24` | Staleness threshold |
| `RETRAIN_MARKETS` | `3000` | Markets in a full retrain |
| `WARMSTART_MARKETS` | `500` | Markets in a warm-start |
| `WARMSTART_TRIGGER_N` | `10` | Resolved bets that trigger a warm-start |

## CI

- **CI** on push and pull request: `cargo fmt --check`, Clippy, tests.
- **Deploy** builds the GHCR images and rsyncs Compose, scripts, migrations, and monitoring to the host in `DEPLOY_HOST`.
- **Release** is a manual workflow that bumps both crate versions, tags, and opens a GitHub release.

## Architecture decision records

| ADR | Title | Status |
|-----|--------|--------|
| [001](adr/001-baseline.md) | Baseline model audit | active |
| [002](adr/002-fix-training-inconsistencies.md) | Training and live feature alignment | complete |
| [003](adr/003-fresh-data-fetch.md) | Fresh fetch and deployment-window filter | complete |
| [004](adr/004-online-learning.md) | Feature store and warm-start | draft |
| [005](adr/005-feature-improvements.md) | Temporal features, target encoding, SHAP | active |
| [006](adr/006-remove-dead-features.md) | Remove dead features | active |
| [007](adr/007-portfolio-correlation-check.md) | LLM correlation check | draft |
| [008](adr/008-workspace-split.md) | Three-crate workspace | accepted |
| [009](adr/009-profitability-overhaul.md) | Signal filters and recalibration | active |
| [010](adr/010-remove-dataset-scripts.md) | Remove external dataset scripts | active |
