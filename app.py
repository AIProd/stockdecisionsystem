import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


APP_TITLE = "Swing Trading Stock Scanner - India"


DEFAULT_UNIVERSE = """
RELIANCE
TCS
HDFCBANK
ICICIBANK
INFY
LT
SBIN
BHARTIARTL
AXISBANK
KOTAKBANK
BAJFINANCE
HINDUNILVR
ITC
MARUTI
TITAN
SUNPHARMA
ULTRACEMCO
ASIANPAINT
WIPRO
TECHM
HCLTECH
KPITTECH
NEWGEN
IIFL
HFCL
SYNGENE
BAJAJ-AUTO
"""


FUNDAMENTAL_ALIASES = {
    "symbol": ["symbol", "ticker", "stock", "name"],
    "market_cap_cr": ["market cap", "market capitalization", "mcap", "market_cap", "market cap cr"],
    "pe": ["pe", "p/e", "price to earning", "price earning", "price to earnings"],
    "pb": ["pb", "p/b", "price to book", "price book"],
    "roe": ["roe", "return on equity"],
    "roce": ["roce", "return on capital employed"],
    "debt_to_equity": ["debt to equity", "debt/equity", "debt equity", "d/e"],
    "sales_growth_3y": ["sales growth 3years", "sales growth 3y", "sales growth", "sales growth 3 years"],
    "profit_growth_3y": ["profit growth 3years", "profit growth 3y", "profit growth", "profit growth 3 years"],
    "pledged_pct": ["pledged percentage", "pledge", "pledged %", "promoter pledge"],
    "promoter_holding": ["promoter holding", "promoters holding", "promoter %"],
}


@dataclass
class ScanConfig:
    min_traded_value_cr: float
    min_volume_mult: float
    min_day_change_pct: float
    max_day_change_pct: float
    require_above_sma20: bool
    require_above_sma50: bool
    require_above_sma200: bool
    require_breakout_20d: bool
    require_potential_breakout: bool
    potential_breakout_distance_pct: float
    min_close_position: float
    max_extension_from_sma20_pct: float
    use_rsi_filter: bool
    min_rsi: float
    max_rsi: float
    use_adx_filter: bool
    min_adx: float
    require_macd_bullish: bool
    require_nr7_or_inside: bool


@dataclass
class FundamentalConfig:
    use_fundamentals: bool
    min_market_cap_cr: float
    max_pe: float
    max_pb: float
    min_roce: float
    min_roe: float
    max_debt_to_equity: float
    min_sales_growth_3y: float
    min_profit_growth_3y: float
    max_pledged_pct: float


@dataclass
class RiskConfig:
    capital: float
    risk_per_trade_pct: float
    atr_stop_mult: float
    target_r_multiple: float


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    symbol = symbol.replace(" ", "")
    if not symbol:
        return ""
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return f"{symbol}.NS"


def display_symbol(symbol: str) -> str:
    return str(symbol).replace(".NS", "").replace(".BO", "")


def parse_symbols(raw_text: str, uploaded_file) -> List[str]:
    symbols: List[str] = []

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        possible_cols = [c for c in df.columns if c.strip().lower() in ["symbol", "ticker", "stock"]]
        if possible_cols:
            symbols.extend(df[possible_cols[0]].dropna().astype(str).tolist())
        else:
            symbols.extend(df.iloc[:, 0].dropna().astype(str).tolist())

    if raw_text:
        for item in raw_text.replace(",", "\n").splitlines():
            item = item.strip()
            if item:
                symbols.append(item)

    clean_symbols = []
    seen = set()
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized and normalized not in seen:
            seen.add(normalized)
            clean_symbols.append(normalized)

    return clean_symbols


def safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return np.nan

    value = str(value).replace(",", "").replace("%", "").replace("₹", "").strip()

    if value in ["", "-", "nan", "None"]:
        return np.nan

    multiplier = 1.0
    lower_value = value.lower()

    if lower_value.endswith("cr"):
        value = value[:-2].strip()
        multiplier = 1.0
    elif lower_value.endswith("crore"):
        value = value[:-5].strip()
        multiplier = 1.0

    try:
        return float(value) * multiplier
    except ValueError:
        return np.nan


def standardize_fundamental_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    original_cols = list(df.columns)
    normalized_lookup = {c.strip().lower(): c for c in original_cols}

    rename_map = {}

    for canonical, aliases in FUNDAMENTAL_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized_lookup:
                rename_map[normalized_lookup[alias.lower()]] = canonical
                break

    df = df.rename(columns=rename_map)

    if "symbol" not in df.columns:
        st.warning("Fundamental CSV does not have a symbol/ticker column. Fundamentals will be ignored.")
        return pd.DataFrame()

    df["symbol"] = df["symbol"].astype(str).map(normalize_symbol)

    numeric_cols = [
        "market_cap_cr",
        "pe",
        "pb",
        "roe",
        "roce",
        "debt_to_equity",
        "sales_growth_3y",
        "profit_growth_3y",
        "pledged_pct",
        "promoter_holding",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(safe_float)

    return df.drop_duplicates("symbol")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(symbols: Tuple[str, ...], period: str) -> pd.DataFrame:
    frames = []

    for symbol in symbols:
        try:
            data = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if data.empty:
                continue

            data = data.reset_index()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in data.columns]
            if missing:
                continue

            data = data[required].copy()
            data["symbol"] = symbol
            frames.append(data)

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.dropna(subset=["Close", "High", "Low", "Open", "Volume"])
    prices = prices.sort_values(["symbol", "Date"])
    return prices


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_smooth
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def add_indicators_one_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    df["prev_close"] = df["Close"].shift(1)
    df["day_change_pct"] = ((df["Close"] - df["prev_close"]) / df["prev_close"]) * 100

    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["sma200"] = df["Close"].rolling(200).mean()

    df["ema9"] = ema(df["Close"], 9)
    df["ema20"] = ema(df["Close"], 20)

    df["avg_vol20"] = df["Volume"].rolling(20).mean()
    df["volume_mult"] = df["Volume"] / df["avg_vol20"]

    df["traded_value_cr"] = (df["Close"] * df["Volume"]) / 10_000_000

    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_position"] = (df["Close"] - df["Low"]) / candle_range

    df["high_20_prev"] = df["High"].shift(1).rolling(20).max()
    df["high_55_prev"] = df["High"].shift(1).rolling(55).max()
    df["high_252"] = df["High"].rolling(252).max()

    df["breakout_20d"] = df["Close"] > df["high_20_prev"]
    df["breakout_55d"] = df["Close"] > df["high_55_prev"]
    df["distance_from_52w_high_pct"] = ((df["Close"] / df["high_252"]) - 1) * 100

    df["rsi14"] = rsi(df["Close"], 14)
    df["atr14"] = atr(df, 14)
    df["adx14"] = adx(df, 14)

    df["macd"] = ema(df["Close"], 12) - ema(df["Close"], 26)
    df["macd_signal"] = ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["daily_range"] = df["High"] - df["Low"]
    df["nr7"] = df["daily_range"] == df["daily_range"].rolling(7).min()

    df["inside_bar"] = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))

    df["extension_from_sma20_pct"] = ((df["Close"] / df["sma20"]) - 1) * 100

    return df


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices

    frames = []
    for _, group in prices.groupby("symbol"):
        frames.append(add_indicators_one_symbol(group))

    return pd.concat(frames, ignore_index=True)


def latest_rows(indicator_df: pd.DataFrame) -> pd.DataFrame:
    if indicator_df.empty:
        return indicator_df

    idx = indicator_df.groupby("symbol")["Date"].idxmax()
    latest = indicator_df.loc[idx].copy()
    latest["display_symbol"] = latest["symbol"].map(display_symbol)
    return latest.sort_values("display_symbol")


def apply_technical_filters(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    out = df.copy()

    out = out[out["traded_value_cr"] >= cfg.min_traded_value_cr]
    out = out[out["volume_mult"] >= cfg.min_volume_mult]
    out = out[out["day_change_pct"] >= cfg.min_day_change_pct]
    out = out[out["day_change_pct"] <= cfg.max_day_change_pct]
    out = out[out["close_position"] >= cfg.min_close_position]
    out = out[out["extension_from_sma20_pct"] <= cfg.max_extension_from_sma20_pct]

    if cfg.require_above_sma20:
        out = out[out["Close"] > out["sma20"]]

    if cfg.require_above_sma50:
        out = out[out["Close"] > out["sma50"]]

    if cfg.require_above_sma200:
        out = out[out["Close"] > out["sma200"]]

    if cfg.require_breakout_20d:
        out = out[out["breakout_20d"] == True]

    if cfg.require_potential_breakout:
        distance_to_20d_high = ((out["high_20_prev"] / out["Close"]) - 1) * 100
        out = out[
            (distance_to_20d_high >= 0)
            & (distance_to_20d_high <= cfg.potential_breakout_distance_pct)
        ]

    if cfg.use_rsi_filter:
        out = out[(out["rsi14"] >= cfg.min_rsi) & (out["rsi14"] <= cfg.max_rsi)]

    if cfg.use_adx_filter:
        out = out[out["adx14"] >= cfg.min_adx]

    if cfg.require_macd_bullish:
        out = out[out["macd_hist"] > 0]

    if cfg.require_nr7_or_inside:
        out = out[(out["nr7"] == True) | (out["inside_bar"] == True)]

    return out


def apply_fundamental_filters(df: pd.DataFrame, cfg: FundamentalConfig) -> pd.DataFrame:
    if not cfg.use_fundamentals:
        return df

    out = df.copy()

    filters = [
        ("market_cap_cr", ">=", cfg.min_market_cap_cr),
        ("pe", "<=", cfg.max_pe),
        ("pb", "<=", cfg.max_pb),
        ("roce", ">=", cfg.min_roce),
        ("roe", ">=", cfg.min_roe),
        ("debt_to_equity", "<=", cfg.max_debt_to_equity),
        ("sales_growth_3y", ">=", cfg.min_sales_growth_3y),
        ("profit_growth_3y", ">=", cfg.min_profit_growth_3y),
        ("pledged_pct", "<=", cfg.max_pledged_pct),
    ]

    for col, operator, threshold in filters:
        if col not in out.columns:
            continue

        if operator == ">=":
            out = out[(out[col].isna()) | (out[col] >= threshold)]
        elif operator == "<=":
            out = out[(out[col].isna()) | (out[col] <= threshold)]

    return out


def calculate_conviction_score(df: pd.DataFrame, use_fundamentals: bool) -> pd.DataFrame:
    out = df.copy()

    score = pd.Series(0, index=out.index, dtype=float)
    max_score = 0

    checks = [
        ("above_sma20", out["Close"] > out["sma20"], 8),
        ("above_sma50", out["Close"] > out["sma50"], 8),
        ("above_sma200", out["Close"] > out["sma200"], 8),
        ("volume_expansion", out["volume_mult"] >= 1.5, 12),
        ("breakout_20d", out["breakout_20d"] == True, 12),
        ("strong_close", out["close_position"] >= 0.65, 10),
        ("rsi_good", out["rsi14"].between(45, 70), 8),
        ("adx_trend", out["adx14"] >= 18, 8),
        ("macd_positive", out["macd_hist"] > 0, 6),
        ("liquid", out["traded_value_cr"] >= 10, 10),
        ("not_overextended", out["extension_from_sma20_pct"] <= 10, 10),
    ]

    tag_values = []

    for label, condition, points in checks:
        max_score += points
        score += np.where(condition.fillna(False), points, 0)
        tag_values.append((label, condition.fillna(False)))

    if use_fundamentals:
        fundamental_checks = []

        if "market_cap_cr" in out.columns:
            fundamental_checks.append(("mcap_ok", out["market_cap_cr"] >= 1000, 6))

        if "pe" in out.columns:
            fundamental_checks.append(("pe_reasonable", out["pe"] <= 50, 5))

        if "roce" in out.columns:
            fundamental_checks.append(("roce_good", out["roce"] >= 15, 8))

        if "roe" in out.columns:
            fundamental_checks.append(("roe_good", out["roe"] >= 12, 6))

        if "debt_to_equity" in out.columns:
            fundamental_checks.append(("low_debt", out["debt_to_equity"] <= 0.8, 8))

        if "pledged_pct" in out.columns:
            fundamental_checks.append(("low_pledge", out["pledged_pct"] <= 5, 8))

        if "sales_growth_3y" in out.columns:
            fundamental_checks.append(("sales_growth", out["sales_growth_3y"] >= 10, 5))

        if "profit_growth_3y" in out.columns:
            fundamental_checks.append(("profit_growth", out["profit_growth_3y"] >= 10, 5))

        for label, condition, points in fundamental_checks:
            max_score += points
            score += np.where(condition.fillna(False), points, 0)
            tag_values.append((label, condition.fillna(False)))

    out["conviction_score"] = (score / max_score * 100).round(1)

    def grade(score_value: float) -> str:
        if score_value >= 85:
            return "A+"
        if score_value >= 75:
            return "A"
        if score_value >= 65:
            return "B"
        if score_value >= 55:
            return "C"
        return "Ignore"

    out["grade"] = out["conviction_score"].map(grade)

    tags = []
    for idx in out.index:
        passed = []
        for label, condition in tag_values:
            try:
                if bool(condition.loc[idx]):
                    passed.append(label)
            except Exception:
                pass
        tags.append(", ".join(passed))

    out["setup_tags"] = tags

    return out


def add_trade_plan(df: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    out = df.copy()

    out["entry"] = out["Close"]
    out["stop"] = out["Close"] - (cfg.atr_stop_mult * out["atr14"])
    out["risk_per_share"] = out["entry"] - out["stop"]
    out["target"] = out["entry"] + (cfg.target_r_multiple * out["risk_per_share"])

    risk_amount = cfg.capital * (cfg.risk_per_trade_pct / 100)
    out["risk_amount"] = risk_amount

    out["qty"] = np.floor(risk_amount / out["risk_per_share"]).replace([np.inf, -np.inf], np.nan)
    out["qty"] = out["qty"].fillna(0).clip(lower=0).astype(int)

    out["capital_required"] = out["qty"] * out["entry"]
    out["planned_risk"] = out["qty"] * out["risk_per_share"]
    out["planned_profit_at_target"] = out["qty"] * (out["target"] - out["entry"])
    out["target_return_pct"] = ((out["target"] / out["entry"]) - 1) * 100
    out["stop_loss_pct"] = ((out["entry"] / out["stop"]) - 1) * 100

    return out


def prepare_output_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "display_symbol",
        "grade",
        "conviction_score",
        "Close",
        "day_change_pct",
        "volume_mult",
        "traded_value_cr",
        "rsi14",
        "adx14",
        "extension_from_sma20_pct",
        "distance_from_52w_high_pct",
        "breakout_20d",
        "nr7",
        "inside_bar",
        "entry",
        "stop",
        "target",
        "target_return_pct",
        "qty",
        "capital_required",
        "planned_risk",
        "planned_profit_at_target",
        "setup_tags",
    ]

    optional_cols = [
        "market_cap_cr",
        "pe",
        "pb",
        "roce",
        "roe",
        "debt_to_equity",
        "sales_growth_3y",
        "profit_growth_3y",
        "pledged_pct",
    ]

    for col in optional_cols:
        if col in df.columns:
            cols.append(col)

    existing_cols = [c for c in cols if c in df.columns]

    out = df[existing_cols].copy()
    out = out.sort_values(["grade", "conviction_score", "traded_value_cr"], ascending=[True, False, False])

    rename_map = {
        "display_symbol": "Symbol",
        "conviction_score": "Conviction Score",
        "Close": "Close",
        "day_change_pct": "Day Change %",
        "volume_mult": "Volume x 20D",
        "traded_value_cr": "Traded Value Cr",
        "rsi14": "RSI 14",
        "adx14": "ADX 14",
        "extension_from_sma20_pct": "Extension From SMA20 %",
        "distance_from_52w_high_pct": "Distance From 52W High %",
        "breakout_20d": "20D Breakout",
        "nr7": "NR7",
        "inside_bar": "Inside Bar",
        "entry": "Entry",
        "stop": "Stop",
        "target": "Target",
        "target_return_pct": "Target Return %",
        "qty": "Qty",
        "capital_required": "Capital Required",
        "planned_risk": "Planned Risk",
        "planned_profit_at_target": "Profit At Target",
        "setup_tags": "Setup Tags",
        "market_cap_cr": "Market Cap Cr",
        "pe": "PE",
        "pb": "PB",
        "roce": "ROCE",
        "roe": "ROE",
        "debt_to_equity": "Debt/Equity",
        "sales_growth_3y": "Sales Growth 3Y",
        "profit_growth_3y": "Profit Growth 3Y",
        "pledged_pct": "Pledged %",
    }

    out = out.rename(columns=rename_map)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(2)

    return out


def plot_stock(symbol: str, indicator_df: pd.DataFrame) -> go.Figure:
    df = indicator_df[indicator_df["symbol"] == symbol].copy().tail(180)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df["Date"], y=df["sma20"], name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["sma50"], name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["sma200"], name="SMA200"), row=1, col=1)

    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume"),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        title=f"{display_symbol(symbol)} - Price, Moving Averages, Volume",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def sidebar_config() -> Tuple[List[str], str, ScanConfig, FundamentalConfig, RiskConfig, Optional[pd.DataFrame]]:
    st.sidebar.header("1. Universe")

    uploaded_universe = st.sidebar.file_uploader(
        "Upload universe CSV",
        type=["csv"],
        help="CSV should have Symbol/Ticker column, or symbols in first column.",
    )

    raw_symbols = st.sidebar.text_area(
        "Or paste symbols",
        value=DEFAULT_UNIVERSE.strip(),
        height=180,
    )

    symbols = parse_symbols(raw_symbols, uploaded_universe)

    period = st.sidebar.selectbox("Price history period", ["6mo", "1y", "2y", "5y"], index=2)

    st.sidebar.header("2. Technical Filters")

    min_traded_value_cr = st.sidebar.number_input("Min traded value ₹ Cr", 0.0, 500.0, 10.0, 1.0)
    min_volume_mult = st.sidebar.slider("Min volume multiple vs 20D avg", 0.5, 5.0, 1.5, 0.1)
    min_day_change_pct = st.sidebar.slider("Min daily change %", -10.0, 20.0, 2.0, 0.5)
    max_day_change_pct = st.sidebar.slider("Max daily change %", 0.0, 30.0, 8.0, 0.5)

    require_above_sma20 = st.sidebar.checkbox("Close > SMA20", True)
    require_above_sma50 = st.sidebar.checkbox("Close > SMA50", True)
    require_above_sma200 = st.sidebar.checkbox("Close > SMA200", True)

    require_breakout_20d = st.sidebar.checkbox("Require 20D breakout", True)
    require_potential_breakout = st.sidebar.checkbox("Potential breakout near 20D high", False)
    potential_breakout_distance_pct = st.sidebar.slider(
        "Potential breakout distance from 20D high %",
        0.5,
        10.0,
        3.0,
        0.5,
    )

    min_close_position = st.sidebar.slider(
        "Close position in candle",
        0.0,
        1.0,
        0.65,
        0.05,
        help="0 = close near low, 1 = close near high.",
    )

    max_extension_from_sma20_pct = st.sidebar.slider(
        "Max extension above SMA20 %",
        2.0,
        40.0,
        12.0,
        0.5,
    )

    use_rsi_filter = st.sidebar.checkbox("Use RSI filter", True)
    min_rsi = st.sidebar.slider("Min RSI", 10.0, 90.0, 45.0, 1.0)
    max_rsi = st.sidebar.slider("Max RSI", 10.0, 95.0, 72.0, 1.0)

    use_adx_filter = st.sidebar.checkbox("Use ADX filter", False)
    min_adx = st.sidebar.slider("Min ADX", 5.0, 60.0, 18.0, 1.0)

    require_macd_bullish = st.sidebar.checkbox("MACD histogram > 0", False)
    require_nr7_or_inside = st.sidebar.checkbox("NR7 or Inside Bar compression", False)

    scan_cfg = ScanConfig(
        min_traded_value_cr=min_traded_value_cr,
        min_volume_mult=min_volume_mult,
        min_day_change_pct=min_day_change_pct,
        max_day_change_pct=max_day_change_pct,
        require_above_sma20=require_above_sma20,
        require_above_sma50=require_above_sma50,
        require_above_sma200=require_above_sma200,
        require_breakout_20d=require_breakout_20d,
        require_potential_breakout=require_potential_breakout,
        potential_breakout_distance_pct=potential_breakout_distance_pct,
        min_close_position=min_close_position,
        max_extension_from_sma20_pct=max_extension_from_sma20_pct,
        use_rsi_filter=use_rsi_filter,
        min_rsi=min_rsi,
        max_rsi=max_rsi,
        use_adx_filter=use_adx_filter,
        min_adx=min_adx,
        require_macd_bullish=require_macd_bullish,
        require_nr7_or_inside=require_nr7_or_inside,
    )

    st.sidebar.header("3. Fundamentals")

    fundamental_file = st.sidebar.file_uploader(
        "Upload fundamentals CSV",
        type=["csv"],
        help="Use export from Screener/Trendlyne/custom sheet. Do not scrape.",
    )

    fundamental_df = None
    use_fundamentals = st.sidebar.checkbox("Use fundamental filters", False)

    min_market_cap_cr = st.sidebar.number_input("Min market cap ₹ Cr", 0.0, 500000.0, 1000.0, 500.0)
    max_pe = st.sidebar.number_input("Max PE", 0.0, 300.0, 60.0, 5.0)
    max_pb = st.sidebar.number_input("Max PB", 0.0, 100.0, 20.0, 1.0)
    min_roce = st.sidebar.number_input("Min ROCE %", -100.0, 100.0, 12.0, 1.0)
    min_roe = st.sidebar.number_input("Min ROE %", -100.0, 100.0, 10.0, 1.0)
    max_debt_to_equity = st.sidebar.number_input("Max Debt/Equity", 0.0, 10.0, 1.0, 0.1)
    min_sales_growth_3y = st.sidebar.number_input("Min Sales Growth 3Y %", -100.0, 300.0, 8.0, 1.0)
    min_profit_growth_3y = st.sidebar.number_input("Min Profit Growth 3Y %", -100.0, 300.0, 8.0, 1.0)
    max_pledged_pct = st.sidebar.number_input("Max pledged %", 0.0, 100.0, 5.0, 1.0)

    if fundamental_file is not None:
        raw_fundamental_df = pd.read_csv(fundamental_file)
        fundamental_df = standardize_fundamental_df(raw_fundamental_df)

    fundamental_cfg = FundamentalConfig(
        use_fundamentals=use_fundamentals,
        min_market_cap_cr=min_market_cap_cr,
        max_pe=max_pe,
        max_pb=max_pb,
        min_roce=min_roce,
        min_roe=min_roe,
        max_debt_to_equity=max_debt_to_equity,
        min_sales_growth_3y=min_sales_growth_3y,
        min_profit_growth_3y=min_profit_growth_3y,
        max_pledged_pct=max_pledged_pct,
    )

    st.sidebar.header("4. Risk")

    capital = st.sidebar.number_input("Total capital ₹", 10000.0, 100000000.0, 3000000.0, 10000.0)
    risk_per_trade_pct = st.sidebar.slider("Risk per trade %", 0.1, 5.0, 0.5, 0.1)
    atr_stop_mult = st.sidebar.slider("ATR stop multiple", 0.5, 5.0, 1.5, 0.1)
    target_r_multiple = st.sidebar.slider("Target R multiple", 1.0, 5.0, 2.5, 0.25)

    risk_cfg = RiskConfig(
        capital=capital,
        risk_per_trade_pct=risk_per_trade_pct,
        atr_stop_mult=atr_stop_mult,
        target_r_multiple=target_r_multiple,
    )

    return symbols, period, scan_cfg, fundamental_cfg, risk_cfg, fundamental_df


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.markdown(
        """
        This app creates a rules-based shortlist for swing trades.  
        It does **not** auto-buy. The final trade needs catalyst review, chart review, and risk approval.
        """
    )

    symbols, period, scan_cfg, fundamental_cfg, risk_cfg, fundamental_df = sidebar_config()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Universe size", len(symbols))
    col_b.metric("Risk per trade", f"₹{risk_cfg.capital * risk_cfg.risk_per_trade_pct / 100:,.0f}")
    col_c.metric("Target multiple", f"{risk_cfg.target_r_multiple}R")

    if len(symbols) == 0:
        st.error("Add symbols or upload a universe CSV.")
        return

    run_scan = st.button("Run scan", type="primary")

    if not run_scan:
        st.info("Set filters in the sidebar, then click Run scan.")
        return

    with st.spinner("Downloading prices and calculating indicators..."):
        prices = fetch_price_data(tuple(symbols), period)

    if prices.empty:
        st.error("No price data found. Check symbols. Use NSE format like RELIANCE or RELIANCE.NS.")
        return

    with st.spinner("Building scanner table..."):
        indicator_df = add_indicators(prices)
        latest = latest_rows(indicator_df)

        if fundamental_df is not None and not fundamental_df.empty:
            latest = latest.merge(fundamental_df, on="symbol", how="left")

        latest = calculate_conviction_score(
            latest,
            use_fundamentals=fundamental_cfg.use_fundamentals and fundamental_df is not None,
        )

        filtered = apply_technical_filters(latest, scan_cfg)
        filtered = apply_fundamental_filters(filtered, fundamental_cfg)
        filtered = add_trade_plan(filtered, risk_cfg)
        output = prepare_output_table(filtered)

    st.subheader("Filtered candidates")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks with data", latest["symbol"].nunique())
    m2.metric("Filtered candidates", len(filtered))
    m3.metric("A/A+ candidates", int((filtered["grade"].isin(["A", "A+"])).sum()) if not filtered.empty else 0)
    m4.metric("Avg conviction", f"{filtered['conviction_score'].mean():.1f}" if not filtered.empty else "-")

    if output.empty:
        st.warning("No candidates found. Relax filters or increase universe.")
        st.write("Suggestion: turn off ADX/MACD/NR7 first, or reduce volume multiple to 1.2.")
        return

    st.dataframe(output, use_container_width=True, height=450)

    csv = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered watchlist CSV",
        data=csv,
        file_name="swing_watchlist.csv",
        mime="text/csv",
    )

    st.subheader("Candidate review")

    selected_display = st.selectbox(
        "Select stock for chart review",
        options=output["Symbol"].tolist(),
    )

    selected_symbol = normalize_symbol(selected_display)
    selected_row = filtered[filtered["symbol"] == selected_symbol].iloc[0]

    left, right = st.columns([2, 1])

    with left:
        st.plotly_chart(plot_stock(selected_symbol, indicator_df), use_container_width=True)

    with right:
        st.markdown("### Trade plan")
        st.write(f"**Symbol:** {display_symbol(selected_symbol)}")
        st.write(f"**Grade:** {selected_row['grade']}")
        st.write(f"**Conviction Score:** {selected_row['conviction_score']:.1f}")
        st.write(f"**Entry:** ₹{selected_row['entry']:.2f}")
        st.write(f"**Stop:** ₹{selected_row['stop']:.2f}")
        st.write(f"**Target:** ₹{selected_row['target']:.2f}")
        st.write(f"**Qty:** {int(selected_row['qty'])}")
        st.write(f"**Capital required:** ₹{selected_row['capital_required']:,.0f}")
        st.write(f"**Planned risk:** ₹{selected_row['planned_risk']:,.0f}")
        st.write(f"**Profit at target:** ₹{selected_row['planned_profit_at_target']:,.0f}")

        st.markdown("### Manual checks before bet")
        st.checkbox("Real catalyst checked on NSE/BSE?", value=False)
        st.checkbox("Sector is supportive?", value=False)
        st.checkbox("Entry is not chased?", value=False)
        st.checkbox("Stop-loss is acceptable?", value=False)
        st.checkbox("Risk-reward is at least 1:2?", value=False)

        st.markdown("### Kill conditions")
        st.write("- Gap-up too much and reverses.")
        st.write("- Breakout level fails.")
        st.write("- Volume disappears after breakout.")
        st.write("- Nifty/sector breaks down.")
        st.write("- Bad corporate/governance news appears.")

    st.subheader("System notes")

    st.markdown(
        """
        **How to use this properly:**

        1. Run after market close.
        2. Shortlist only A/A+ candidates.
        3. Manually check catalyst.
        4. Do not buy at open blindly.
        5. Prefer retest/pullback entry.
        6. Use position sizing from the app.
        7. Export watchlist and journal every trade.

        **Bad use:** buying every filtered stock.  
        **Good use:** using the scanner as a candidate generator, then applying human judgment.
        """
    )


if __name__ == "__main__":
    main()
