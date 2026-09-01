import os, json, requests, time, re, random, traceback, uuid
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
try:
    import yfinance as yf
except Exception:
    yf = None

# ── News Parsing Helpers ─────────────────────────────────────────────────────
def _coerce_impact(impact: Any) -> str:
    if impact is None:
        return "UNKNOWN"
    if isinstance(impact, (int, float)):
        return "HIGH" if int(impact) >= 3 else "MEDIUM" if int(impact) >= 2 else "LOW"
    text = str(impact).strip().lower()
    if text in {"3", "high", "high impact", "red", "important"}:
        return "HIGH"
    if text in {"2", "medium", "orange", "moderate"}:
        return "MEDIUM"
    return text.upper() if text else "UNKNOWN"

def normalize_event_time(date_str, time_str, timezone_name, reference_dt=None):
    if not date_str:
        return None
    try:
        if isinstance(date_str, datetime):
            event_dt = date_str
        else:
            text = str(date_str).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            event_dt = datetime.fromisoformat(text)
            if event_dt.tzinfo is None:
                if timezone_name:
                    try:
                        event_dt = event_dt.replace(tzinfo=ZoneInfo(str(timezone_name)))
                    except Exception:
                        event_dt = event_dt.replace(tzinfo=timezone.utc)
                else:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
            return event_dt.astimezone(timezone.utc)
    except Exception:
        pass
    if not time_str:
        return None
    try:
        event_date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except ValueError:
        try:
            event_date = datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S").date()
        except ValueError:
            return None
    raw_time = str(time_str).strip()
    try:
        hour, minute = map(int, raw_time.split(":")[:2])
    except ValueError:
        return None
    tzinfo = None
    if timezone_name:
        try:
            tzinfo = ZoneInfo(str(timezone_name))
        except Exception:
            tzinfo = None
    event_dt = datetime(event_date.year, event_date.month, event_date.day, hour, minute, tzinfo=tzinfo)
    if tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    if reference_dt is not None and reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)
    return event_dt.astimezone(timezone.utc)

def parse_news_payload(payload, reference_dt=None, lookahead_hours=72, only_high=True):
    if reference_dt is None:
        reference_dt = datetime.now(timezone.utc)
    if reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)
    events = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        impact = _coerce_impact(item.get("impact"))
        if only_high and impact != "HIGH":
            continue
        if not only_high and impact not in {"HIGH", "MEDIUM"}:
            continue
        event_dt = normalize_event_time(item.get("date"), item.get("time"), item.get("timezone") or item.get("tz") or item.get("timeZone"), reference_dt=reference_dt)
        if event_dt is None:
            continue
        if event_dt < reference_dt - timedelta(hours=6):
            continue
        if event_dt > reference_dt + timedelta(hours=lookahead_hours):
            continue
        minutes_until = int((event_dt - reference_dt).total_seconds() // 60)
        events.append({"event": str(item.get("event") or item.get("title") or "Economic Event").strip(), "currency": str(item.get("country") or item.get("currency") or item.get("pair") or "USD").strip(), "impact": "HIGH" if impact == "HIGH" else "MEDIUM", "time": event_dt.strftime("%Y-%m-%d %H:%M UTC"), "event_time_utc": event_dt, "minutes_until": minutes_until, "within_2h": minutes_until <= 120 and minutes_until >= 0, "timezone": item.get("timezone") or item.get("tz") or item.get("timeZone") or "UTC"})
    events.sort(key=lambda e: e["event_time_utc"])
    return events

def build_news_context(events, reference_dt=None):
    if reference_dt is None:
        reference_dt = datetime.now(timezone.utc)
    if reference_dt.tzinfo is None:
        reference_dt = reference_dt.replace(tzinfo=timezone.utc)
    upcoming = [e for e in events if e.get("event_time_utc") and e["event_time_utc"] >= reference_dt]
    within_2h = [e for e in upcoming if e.get("within_2h")]
    next_event = upcoming[0] if upcoming else None
    if next_event:
        minutes_until = int((next_event["event_time_utc"] - reference_dt).total_seconds() // 60)
        if minutes_until <= 120:
            bias = "opposite"
            pre_news_bias = f"High-impact event arriving in {minutes_until} minutes; expect the market to express a short-term reactive move before stabilizing."
        else:
            bias = "neutral"
            pre_news_bias = f"Upcoming high-impact event in {minutes_until} minutes; monitor for volatility expansion and a likely liquidity sweep."
    else:
        bias = "neutral"
        pre_news_bias = "No imminent high-impact event in the next 2 hours."
    return {"within_2h": bool(within_2h), "bias": bias, "upcoming_count": len(upcoming), "next_event": next_event, "pre_news_bias": pre_news_bias, "summary": "\n".join([f"- {e['time']} | {e['currency']} | {e['event']}" for e in upcoming[:5]])}

def format_news_summary(events, limit=5):
    if not events:
        return "No high-impact news in the upcoming window."
    items = events[:limit]
    return "\n".join([f"- {e['time']} {e['currency']}: {e['event']}" for e in items])

def format_east_africa_time(dt):
    try:
        east_africa = dt.astimezone(ZoneInfo("Africa/Nairobi"))
        return east_africa.strftime("%Y-%m-%d %H:%M EAT")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M UTC")

def is_same_day_event(event_dt, reference_dt=None):
    reference = reference_dt or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    try:
        east_ref = reference.astimezone(ZoneInfo("Africa/Nairobi"))
        east_event = event_dt.astimezone(ZoneInfo("Africa/Nairobi"))
        return east_ref.date() == east_event.date()
    except Exception:
        return event_dt.date() == reference.date()

def is_usd_sensitive_news(event):
    currency = str(event.get('currency') or '').upper()
    event_name = str(event.get('event') or '').upper()
    if currency == 'USD':
        return True
    usd_keywords = ['FED', 'FOMC', 'CPI', 'PPI', 'NFP', 'PAYROLL', 'UNEMPLOYMENT', 'JOBLESS', 'RETAIL SALES', 'GDP', 'ISM', 'PMI', 'CONSUMER CONFIDENCE', 'TREASURY', 'INFLATION', 'PCE', 'JOLTS', 'CONSTRUCTION', 'HOME SALES', 'TRADE BALANCE', 'DURABLE GOODS', 'MICHIGAN', 'FEDERAL RESERVE', 'DOLLAR', 'USD']
    return any(keyword in event_name for keyword in usd_keywords)

# FIX: Allow MEDIUM impact events if they are USD-sensitive (e.g., Jobless Claims)
def filter_relevant_news(events, selected_symbols=None, reference_dt=None):
    reference = reference_dt or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    filtered = []

    for event in events:
        impact = str(event.get('impact') or '').upper()
        currency = str(event.get('currency') or '').upper()

        # Allow only HIGH and MEDIUM impact
        if impact not in ('HIGH', 'MEDIUM'):
            continue

        # STRICT: allow only USD news for both HIGH and MEDIUM impact
        if currency != 'USD':
            continue

        event_dt = event.get('event_time_utc')
        if event_dt is None:
            continue

        if event_dt < reference - timedelta(hours=6):
            continue

        if event_dt > reference + timedelta(hours=24):
            continue

        if not is_same_day_event(event_dt, reference):
            continue

        filtered.append(event)

    filtered.sort(key=lambda e: e['event_time_utc'])
    return filtered

def calculate_rsi(series, period=14):
    if series is None:
        return pd.Series(dtype=float)
    original = pd.Series(series)
    if len(original) < 2:
        return pd.Series([np.nan] * len(original), index=original.index, dtype=float)
    series = pd.to_numeric(original, errors='coerce').dropna()
    if series.empty:
        return pd.Series([np.nan] * len(original), index=original.index, dtype=float)
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    window = min(period, len(series))
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(original.index, fill_value=50).astype(float)

def detect_rsi_divergence(df, period=14):
    if df is None or df.empty:
        return None
    closes = pd.to_numeric(df['Close'], errors='coerce').dropna()
    if closes.empty or len(closes) < 4:
        return None
    rsi = calculate_rsi(closes, period=period)
    if rsi.empty:
        return None
    pivot_highs = []
    pivot_lows = []
    for i in range(2, len(closes) - 2):
        prev2 = closes.iloc[i - 2]
        prev1 = closes.iloc[i - 1]
        curr = closes.iloc[i]
        next1 = closes.iloc[i + 1]
        next2 = closes.iloc[i + 2]
        if curr >= prev1 and curr >= next1 and curr >= prev2 and curr >= next2:
            pivot_highs.append((i, float(curr), float(rsi.iloc[i])))
        if curr <= prev1 and curr <= next1 and curr <= prev2 and curr <= next2:
            pivot_lows.append((i, float(curr), float(rsi.iloc[i])))
    if len(pivot_highs) >= 2:
        prev_high = pivot_highs[-2]
        curr_high = pivot_highs[-1]
        if curr_high[1] < prev_high[1] and curr_high[2] > prev_high[2]:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is printing a lower high while RSI is holding a higher high, suggesting bullish divergence.'}
        if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is printing a higher high while RSI is failing, suggesting bearish divergence.'}
    if len(pivot_lows) >= 2:
        prev_low = pivot_lows[-2]
        curr_low = pivot_lows[-1]
        if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is printing a lower low while RSI is holding a higher low, suggesting bullish divergence.'}
        if curr_low[1] > prev_low[1] and curr_low[2] < prev_low[2]:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is printing a higher low while RSI is weakening, suggesting bearish divergence.'}
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])
    last_rsi = float(rsi.iloc[-1])
    prev_rsi = float(rsi.iloc[-2])
    if last_close < prev_close and last_rsi > prev_rsi:
        return {'type': 'BULLISH_DIV', 'reason': 'The latest candles show a bullish RSI divergence against the recent price decline.'}
    if last_close > prev_close and last_rsi < prev_rsi:
        return {'type': 'BEARISH_DIV', 'reason': 'The latest candles show a bearish RSI divergence against the recent price advance.'}
    if len(closes) >= 6:
        lookback = min(5, len(closes) - 1)
        price_change = float(closes.iloc[-1] - closes.iloc[-lookback - 1])
        rsi_change = float(rsi.iloc[-1] - rsi.iloc[-lookback - 1])
        if price_change <= 0 and rsi_change > 0:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is failing to continue lower while RSI is rising, which is bullish divergence.'}
        if price_change >= 0 and rsi_change < 0:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is pushing higher while RSI is weakening, which is bearish divergence.'}
    if len(closes) >= 4:
        prev_prices = closes.iloc[:-1]
        prev_rsi = rsi.iloc[:-1]
        prev_high = float(prev_prices.max())
        prev_low = float(prev_prices.min())
        prev_rsi_high = float(prev_rsi.max())
        prev_rsi_low = float(prev_rsi.min())
        if last_close > prev_high and last_rsi < prev_rsi_high:
            return {'type': 'BULLISH_DIV', 'reason': 'Price is making a new high while RSI is failing to confirm, which often points to a bullish divergence.'}
        if last_close < prev_low and last_rsi > prev_rsi_low:
            return {'type': 'BEARISH_DIV', 'reason': 'Price is making a new low while RSI is failing to confirm, which often points to a bearish divergence.'}
    if len(closes) >= 4 and last_close < closes.iloc[-3] and last_rsi > rsi.iloc[-3]:
        return {'type': 'BULLISH_DIV', 'reason': 'Price is failing to make a new low while RSI is rising, which is classic bullish divergence.'}
    if len(closes) >= 4 and last_close > closes.iloc[-3] and last_rsi < rsi.iloc[-3]:
        return {'type': 'BEARISH_DIV', 'reason': 'Price is pushing higher while RSI is weakening, which is classic bearish divergence.'}
    return None

def build_market_structure_summary(df, current_price=None, swings=None, order_blocks=None, fvgs=None, sweeps=None, bos=None, choch=None, symbol=None):
    if df is None or df.empty:
        return 'Market structure unavailable.'
    current_price = float(current_price) if current_price is not None else float(df['Close'].iloc[-1])
    swings = swings or find_swings(df)
    recent_swing_highs = [float(v) for v in swings.get('recent_swing_highs', []) if v is not None]
    recent_swing_lows = [float(v) for v in swings.get('recent_swing_lows', []) if v is not None]
    zone_buffer = max(abs(current_price) * 0.0015, 0.5 if current_price >= 100 else 0.01)
    support_zones = [f"{max(level - zone_buffer, 0):.2f}-{level + zone_buffer:.2f}" for level in recent_swing_lows[:3]]
    resistance_zones = [f"{level - zone_buffer:.2f}-{level + zone_buffer:.2f}" for level in recent_swing_highs[:3]]
    demand_zones = []
    supply_zones = []
    for ob in (order_blocks or [])[:3]:
        if ob['type'].startswith('BULLISH'):
            demand_zones.append(f"{ob['price'] - zone_buffer:.2f}-{ob['price'] + zone_buffer:.2f}")
        else:
            supply_zones.append(f"{ob['price'] - zone_buffer:.2f}-{ob['price'] + zone_buffer:.2f}")
    for fvg in (fvgs or [])[:2]:
        if fvg['type'].startswith('BULLISH'):
            demand_zones.append(f"{fvg['bottom']:.2f}-{fvg['top']:.2f}")
        else:
            supply_zones.append(f"{fvg['bottom']:.2f}-{fvg['top']:.2f}")
    for sweep in (sweeps or [])[:2]:
        if sweep['type'].startswith('BULLISH'):
            demand_zones.append(f"{sweep['price'] - zone_buffer:.2f}-{sweep['price'] + zone_buffer:.2f}")
        else:
            supply_zones.append(f"{sweep['price'] - zone_buffer:.2f}-{sweep['price'] + zone_buffer:.2f}")
    structure_markers = []
    if bos:
        structure_markers.append(bos)
    if choch:
        structure_markers.append(choch)
    structure_markers = structure_markers[:2]
    support_text = ', '.join(support_zones) if support_zones else 'none'
    resistance_text = ', '.join(resistance_zones) if resistance_zones else 'none'
    demand_text = ', '.join(demand_zones) if demand_zones else 'none'
    supply_text = ', '.join(supply_zones) if supply_zones else 'none'
    marker_text = ', '.join(structure_markers) if structure_markers else 'balanced'
    return (f"Market structure zones: support={support_text}; resistance={resistance_text}; "
            f"demand={demand_text}; supply={supply_text}; structure markers={marker_text}; "
            f"current price={current_price:.2f}. The AI must identify whether price is approaching a demand/supply or support/resistance zone, "
            f"and whether the next move is a continuation or a reversal into the next clear invalidation zone.")

def build_multitimeframe_context(all_data, symbol):
    if not all_data or not symbol:
        return 'Multi-timeframe context unavailable.'
    data = all_data.get(symbol, {})
    if not isinstance(data, dict):
        return 'Multi-timeframe context unavailable.'
    frames = [('10M', data.get('M10')), ('15M', data.get('M15')), ('30M', data.get('M30')), ('1H', data.get('H1')), ('4H', data.get('H4'))]
    parts = []
    for label, frame in frames:
        if frame is None or getattr(frame, 'empty', True):
            continue
        micro = calculate_microstructure(frame)
        if not micro:
            micro = {'price_vs_vwap': 'NEUTRAL', 'momentum': 'NEUTRAL', 'rvol': 1.0}
        bos, choch = detect_bos_choch(frame)
        order_blocks = detect_order_blocks(frame)
        fvgs = detect_fvg(frame)
        sweeps = detect_liquidity_sweeps(frame)
        divergence = detect_rsi_divergence(frame)
        structure_bits = []
        if bos:
            structure_bits.append(bos)
        if choch:
            structure_bits.append(choch)
        if order_blocks:
            structure_bits.append(f"OB:{order_blocks[-1]['type']}")
        if fvgs:
            structure_bits.append(f"FVG:{fvgs[-1]['type']}")
        if sweeps:
            structure_bits.append(f"SWP:{sweeps[-1]['type']}")
        if divergence:
            structure_bits.append(divergence['type'])
        structure_text = ', '.join(structure_bits) if structure_bits else 'balanced structure'
        parts.append(f"{label}: price is {micro['price_vs_vwap']} VWAP, momentum is {micro['momentum']}, RVOL is {micro['rvol']:.2f}, and the current read is {structure_text}.")
    if not parts:
        return 'No usable higher-timeframe or lower-timeframe context is available.'
    return ' '.join(parts)

def build_historical_context(m10):
    if m10 is None or m10.empty:
        return 'Historical context unavailable.'
    closes = [round(float(v), 4) for v in m10['Close'].tail(20).tolist()]
    if len(closes) < 2:
        return 'Historical context unavailable.'
    latest = float(m10['Close'].iloc[-1])
    prev = float(m10['Close'].iloc[-2])
    change_pct = round(((latest - prev) / prev) * 100, 2) if prev else 0.0
    high = float(m10['High'].tail(20).max())
    low = float(m10['Low'].tail(20).min())
    range_pct = round(((high - low) / latest) * 100, 2) if latest else 0.0
    return f"Last 20 closes: {closes}; latest 1-bar change: {change_pct}%; recent 20-bar range: {range_pct}%."

def compute_rsi_last(series, period=14):
    rsi = calculate_rsi(series, period)
    if rsi is None or rsi.empty:
        return None
    val = rsi.dropna()
    if val.empty:
        return None
    return round(float(val.iloc[-1]), 1)

def build_rsi_values_context(all_data, symbol):
    data = all_data.get(symbol, {}) or {}
    parts = []
    for label, key in [('M10', 'M10'), ('M15', 'M15'), ('M30', 'M30'), ('H1', 'H1'), ('H4', 'H4')]:
        df = data.get(key)
        if df is None or getattr(df, 'empty', True):
            continue
        v = compute_rsi_last(pd.to_numeric(df['Close'], errors='coerce'))
        if v is not None:
            state = 'OVERBOUGHT' if v >= 70 else 'OVERSOLD' if v <= 30 else 'neutral'
            parts.append(f"{label} RSI {v} ({state})")
    return ' | '.join(parts) if parts else 'RSI values unavailable.'

def range_position(df, current_price):
    if df is None or df.empty:
        return None
    look = df.tail(60)
    hi = float(look['High'].max())
    lo = float(look['Low'].min())
    if hi <= lo:
        return None
    return (float(current_price) - lo) / (hi - lo), lo, hi

def build_premium_discount_context(m10, current_price):
    pos_info = range_position(m10, current_price)
    if not pos_info:
        return 'Range position unavailable.'
    pos, lo, hi = pos_info
    zone = 'PREMIUM (upper half of range - SMC favors sells/shorts here)' if pos >= 0.5 else 'DISCOUNT (lower half of range - SMC favors buys/longs here)'
    return f"Recent 60-bar range {lo:.2f}-{hi:.2f}; price at {pos * 100:.0f}% of the range => {zone}."

def build_volatility_context(m10):
    atr = calculate_atr(m10)
    if atr is None or m10 is None or m10.empty:
        return 'ATR unavailable.'
    last = float(m10['Close'].iloc[-1])
    atr_pct = atr / last * 100 if last else 0.0
    return f"M10 ATR {atr:.2f} ({atr_pct:.2f}% of price). Judge whether volatility is expanded or compressed versus recent sessions."

def _request_json(url, timeout=15):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def get_live_price_for_symbol(symbol):
    if symbol == 'BTCUSD':
        data = _request_json('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT')
        if data and data.get('price') is not None:
            return float(data['price'])
    elif symbol == 'EURUSD':
        data = _request_json('https://api.frankfurter.app/latest?from=EUR&to=USD')
        if data and data.get('rates', {}).get('USD') is not None:
            return float(data['rates']['USD'])
    elif symbol == 'XAUUSD':
        try:
            response = requests.get('https://api.metals.live/v1/spot/XAUUSD', timeout=15, verify=False)
            response.raise_for_status()
            data = response.json()
            price = data.get('price') if isinstance(data, dict) else None
            if price is not None:
                return float(price)
        except Exception:
            pass
    return None

def get_latest_quote(symbol):
    price = get_live_price_for_symbol(symbol)
    if price is not None:
        return price
    if yf is None:
        return None
    try:
        ticker = yf.Ticker(symbol)
        for attr in ['lastPrice', 'last_price', 'regularMarketPrice', 'currentPrice']:
            value = getattr(getattr(ticker, 'fast_info', None), attr, None)
            if value is not None:
                return float(value)
        hist = ticker.history(period='1d', interval='1m', timeout=20)
        if hist is not None and not hist.empty:
            last_close = hist['Close'].dropna()
            if not last_close.empty:
                return float(last_close.iloc[-1])
    except Exception as exc:
        print(f"⚠️ Quote fetch failed for {symbol}: {exc}")
    return None

def get_live_market_snapshot(symbol, yf_symbol, fallback_df=None):
    """Live quote is used ONLY if it agrees with the analyzed candles (<=0.5% deviation)."""
    fallback_price = None
    if fallback_df is not None and not fallback_df.empty:
        fallback_price = float(fallback_df['Close'].iloc[-1])
    price = get_latest_quote(symbol)
    if price is not None and fallback_price is not None and fallback_price > 0:
        if abs(price - fallback_price) / fallback_price > 0.005:
            price = None
    if price is None:
        price = fallback_price
    return {'symbol': symbol, 'price': price, 'source': 'live' if price is not None and fallback_price is not None and price != fallback_price else 'fallback'}

def calculate_atr(df, period=14):
    if len(df) < period + 2:
        return None
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None

def detect_exhaustion(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    total_range = last['High'] - last['Low']
    if total_range == 0:
        return None
    wick = max(last['High'] - max(last['Open'], last['Close']), min(last['Open'], last['Close']) - last['Low'])
    wick_ratio = wick / total_range
    body_ratio = body / total_range
    vol = float(last['Volume']) if 'Volume' in df.columns else 0.0
    avg_vol = float(df['Volume'].tail(10).mean()) if 'Volume' in df.columns else 0.0
    volume_spike = vol / avg_vol if avg_vol > 0 else 1.0
    bullish_exhaustion = last['Close'] > prev['Close'] and wick_ratio > 0.55 and body_ratio < 0.35 and volume_spike > 1.1
    bearish_exhaustion = last['Close'] < prev['Close'] and wick_ratio > 0.55 and body_ratio < 0.35 and volume_spike > 1.1
    if bullish_exhaustion or bearish_exhaustion:
        return {'type': 'BULLISH_EXHAUSTION' if bullish_exhaustion else 'BEARISH_EXHAUSTION', 'direction': None, 'strength': 'STRONG' if volume_spike > 1.3 else 'MODERATE', 'reason': 'Price made a stretched wick into the current bar and closed with a weak body, signaling exhaustion.'}
    return None

def detect_reversal(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    body = abs(last['Close'] - last['Open'])
    total_range = last['High'] - last['Low']
    if total_range == 0:
        return None
    wick_ratio = max(last['High'] - max(last['Open'], last['Close']), min(last['Open'], last['Close']) - last['Low']) / total_range
    body_ratio = body / total_range
    bullish_reversal = last['Close'] > last['Open'] and last['Close'] >= prev['Close'] and wick_ratio > 0.45 and body_ratio < 0.45 and last['Low'] <= min(prev['Low'], prev2['Low'])
    bearish_reversal = last['Close'] < last['Open'] and last['Close'] <= prev['Close'] and wick_ratio > 0.45 and body_ratio < 0.45 and last['High'] >= max(prev['High'], prev2['High'])
    if bullish_reversal or bearish_reversal:
        return {'type': 'BULLISH_REVERSAL' if bullish_reversal else 'BEARISH_REVERSAL', 'direction': 'BUY' if bullish_reversal else 'SELL', 'strength': 'STRONG' if wick_ratio > 0.6 else 'MODERATE', 'reason': 'The market is rejecting a previous extreme and showing a clean reversal candle.'}
    return None

def detect_continuation(df):
    if len(df) < 8:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    recent_high = max(df['High'].tail(4).iloc[:-1])
    recent_low = min(df['Low'].tail(4).iloc[:-1])
    bullish_continuation = last['Close'] > prev['Close'] and last['Close'] > recent_high and last['Low'] > recent_low
    bearish_continuation = last['Close'] < prev['Close'] and last['Close'] < recent_low and last['High'] < recent_high
    if bullish_continuation or bearish_continuation:
        return {'type': 'BULLISH_CONTINUATION' if bullish_continuation else 'BEARISH_CONTINUATION', 'direction': 'BUY' if bullish_continuation else 'SELL', 'strength': 'STRONG' if abs(last['Close'] - prev['Close']) > (last['High'] - last['Low']) * 0.5 else 'MODERATE', 'reason': 'Price is extending beyond the recent range with follow-through, supporting continuation.'}
    return None

def assess_entry_quality(df, swings, current_price, signal):
    if not swings:
        return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
    if signal == 'BUY':
        swing_lows = [level for level in swings.get('recent_swing_lows', []) if level < current_price]
        zone = swing_lows[0] if swing_lows else None
        if zone is None:
            return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
        distance_pct = abs(current_price - zone) / current_price
        quality = 'early' if distance_pct < 0.003 else 'acceptable' if distance_pct < 0.008 else 'late'
        return {'entry_quality': quality, 'entry_zone': zone, 'distance_pct': round(distance_pct, 6)}
    swing_highs = [level for level in swings.get('recent_swing_highs', []) if level > current_price]
    zone = swing_highs[0] if swing_highs else None
    if zone is None:
        return {'entry_quality': 'unknown', 'entry_zone': None, 'distance_pct': None}
    distance_pct = abs(zone - current_price) / current_price
    quality = 'early' if distance_pct < 0.003 else 'acceptable' if distance_pct < 0.008 else 'late'
    return {'entry_quality': quality, 'entry_zone': zone, 'distance_pct': round(distance_pct, 6)}

def detect_market_phase(df, swings=None):
    micro = calculate_microstructure(df)
    continuation = detect_continuation(df)
    reversal = detect_reversal(df)
    exhaustion = detect_exhaustion(df)
    divergence = detect_rsi_divergence(df)
    entry_quality = assess_entry_quality(df, swings, df['Close'].iloc[-1], 'BUY' if micro.get('momentum') == 'BULLISH' else 'SELL') if swings is not None else {'entry_quality': 'unknown'}
    direction = None
    if continuation:
        phase = 'continuation'
        reason = continuation['reason']
        direction = continuation.get('direction')
    elif reversal:
        phase = 'reversal'
        reason = reversal['reason']
        direction = reversal.get('direction')
    elif exhaustion:
        phase = 'exhaustion'
        reason = exhaustion['reason']
    elif micro.get('momentum') == 'BULLISH' and micro.get('price_vs_vwap') == 'ABOVE':
        phase = 'trend'
        reason = 'Price is holding above VWAP with positive momentum and is still in a directional trend.'
        direction = 'BUY'
    elif micro.get('momentum') == 'BEARISH' and micro.get('price_vs_vwap') == 'BELOW':
        phase = 'trend'
        reason = 'Price is holding below VWAP with negative momentum and is still in a directional trend.'
        direction = 'SELL'
    else:
        phase = 'coiling'
        reason = 'The market is coiling and lacks a clear continuation or reversal edge yet.'
        if divergence:
            direction = 'BUY' if divergence['type'] == 'BULLISH_DIV' else 'SELL'
            reason += f" RSI divergence hint: {divergence['reason']}"
    return {'phase': phase, 'reason': reason, 'direction': direction, 'entry_quality': entry_quality.get('entry_quality', 'unknown'), 'entry_zone': entry_quality.get('entry_zone'), 'distance_pct': entry_quality.get('distance_pct')}

def detect_directional_confluence(df, swings=None, htf_context=None, dxy_context=None, symbol=None):
    if df is None or df.empty:
        return {'direction': None, 'bullish_evidence': [], 'bearish_evidence': [], 'bull_count': 0, 'bear_count': 0}
    bullish_evidence = []
    bearish_evidence = []
    micro = calculate_microstructure(df)
    reversal = detect_reversal(df)
    continuation = detect_continuation(df)
    sweeps = detect_liquidity_sweeps(df)
    divergence = detect_rsi_divergence(df)
    if reversal:
        if reversal.get('direction') == 'BUY':
            bullish_evidence.append(f"reversal candle rejecting lows ({reversal['type']})")
        else:
            bearish_evidence.append(f"reversal candle rejecting highs ({reversal['type']})")
    if continuation:
        if continuation.get('direction') == 'BUY':
            bullish_evidence.append(f"continuation break to the upside ({continuation['type']})")
        else:
            bearish_evidence.append(f"continuation break to the downside ({continuation['type']})")
    for sweep in sweeps[-2:]:
        if sweep['type'] == 'BULLISH_SWEEP':
            bullish_evidence.append('liquidity sweep of lows reclaimed')
        else:
            bearish_evidence.append('liquidity sweep of highs rejected')
    if divergence:
        if divergence['type'] == 'BULLISH_DIV':
            bullish_evidence.append('bullish RSI divergence')
        else:
            bearish_evidence.append('bearish RSI divergence')
    if micro.get('momentum') == 'BULLISH' and micro.get('price_vs_vwap') == 'ABOVE':
        bullish_evidence.append('price holding above VWAP with bullish momentum')
    elif micro.get('momentum') == 'BEARISH' and micro.get('price_vs_vwap') == 'BELOW':
        bearish_evidence.append('price holding below VWAP with bearish momentum')
    pos_info = range_position(df, df['Close'].iloc[-1])
    if pos_info:
        pos, lo, hi = pos_info
        if pos <= 0.35:
            bullish_evidence.append('price in discount zone of the recent range (favors longs)')
        elif pos >= 0.65:
            bearish_evidence.append('price in premium zone of the recent range (favors shorts)')
    if isinstance(htf_context, dict):
        htf_trend = str(htf_context.get('trend') or htf_context.get('bias') or '').upper()
        if htf_trend == 'BULLISH':
            bullish_evidence.append('higher-timeframe trend bullish')
        elif htf_trend == 'BEARISH':
            bearish_evidence.append('higher-timeframe trend bearish')
    if dxy_context and symbol in ['XAUUSD', 'EURUSD', 'BTCUSD']:
        trend = dxy_context.get('trend')
        pos = dxy_context.get('price_vs_vwap')
        if trend == 'BEARISH' and pos == 'BELOW':
            bullish_evidence.append('DXY weakness supporting longs')
        elif trend == 'BULLISH' and pos == 'ABOVE':
            bearish_evidence.append('DXY strength supporting shorts')
    bull = len(bullish_evidence)
    bear = len(bearish_evidence)
    direction = None
    if bull >= 2 and bull > bear:
        direction = 'BUY'
    elif bear >= 2 and bear > bull:
        direction = 'SELL'
    return {'direction': direction, 'bullish_evidence': bullish_evidence, 'bearish_evidence': bearish_evidence, 'bull_count': bull, 'bear_count': bear}

# ── Multi-Timeframe Desk Picture + Firm Bias + Adaptive Learning ─────────────
def build_mtf_picture(all_data, symbol):
    data = all_data.get(symbol, {}) or {}
    specs = [('M10', 1.0), ('M15', 1.0), ('M30', 1.5), ('H1', 2.0), ('H4', 3.0)]
    snaps = []
    for label, weight in specs:
        df = data.get(label)
        if df is None or getattr(df, 'empty', True):
            continue
        micro = calculate_microstructure(df)
        bos, choch = detect_bos_choch(df)
        sweeps = detect_liquidity_sweeps(df)
        divergence = detect_rsi_divergence(df)
        reversal = detect_reversal(df)
        continuation = detect_continuation(df)
        snaps.append({'label': label, 'weight': weight, 'micro': micro, 'bos': bos, 'choch': choch, 'sweeps': sweeps, 'divergence': divergence, 'reversal': reversal, 'continuation': continuation})
    score = 0.0
    bull_items, bear_items = [], []
    for s in snaps:
        w = s['weight']
        m = s['micro']
        if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
            score += w; bull_items.append(f"{s['label']} trend bullish (above VWAP)")
        elif m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
            score -= w; bear_items.append(f"{s['label']} trend bearish (below VWAP)")
        if s['bos'] == 'BULLISH_BOS' or s['choch'] == 'BULLISH_CHOCH':
            score += w; bull_items.append(f"{s['label']} {s['bos'] or s['choch']}")
        elif s['bos'] == 'BEARISH_BOS' or s['choch'] == 'BEARISH_CHOCH':
            score -= w; bear_items.append(f"{s['label']} {s['bos'] or s['choch']}")
        for sw in s['sweeps'][-1:]:
            if sw['type'] == 'BULLISH_SWEEP':
                score += 0.5 * w; bull_items.append(f"{s['label']} swept lows reclaimed")
            else:
                score -= 0.5 * w; bear_items.append(f"{s['label']} swept highs rejected")
        if s['divergence']:
            if s['divergence']['type'] == 'BULLISH_DIV':
                score += 0.5 * w; bull_items.append(f"{s['label']} bullish RSI divergence")
            else:
                score -= 0.5 * w; bear_items.append(f"{s['label']} bearish RSI divergence")
        if s['reversal']:
            if s['reversal']['direction'] == 'BUY':
                score += w; bull_items.append(f"{s['label']} reversal candle rejecting lows")
            else:
                score -= w; bear_items.append(f"{s['label']} reversal candle rejecting highs")
        if s['continuation']:
            if s['continuation']['direction'] == 'BUY':
                score += w; bull_items.append(f"{s['label']} continuation break upside")
            else:
                score -= w; bear_items.append(f"{s['label']} continuation break downside")
    # FIX: stable HTF bias - momentum AND VWAP side must agree on H1/H4.
    htf_reads = []
    for s in snaps:
        if s['label'] in ('H1', 'H4'):
            m = s['micro']
            if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
                htf_reads.append('BULLISH')
            elif m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
                htf_reads.append('BEARISH')
            else:
                htf_reads.append('NEUTRAL')
    htf_bias = 'BULLISH' if htf_reads and all(r == 'BULLISH' for r in htf_reads) else ('BEARISH' if htf_reads and all(r == 'BEARISH' for r in htf_reads) else 'NEUTRAL')
    return {'snaps': snaps, 'score': score, 'bull_items': bull_items, 'bear_items': bear_items, 'htf_bias': htf_bias}

BIAS_MIN_HOLD_MINUTES = 45

# FIX: strong hysteresis - a standing bias only flips on a real H1/H4 structural break.
def resolve_firm_direction(symbol, picture):
    score = picture.get('score', 0)
    htf = picture.get('htf_bias', 'NEUTRAL')
    notes = []
    base = 'BUY' if score >= 2.0 else ('SELL' if score <= -2.0 else None)
    if base and htf != 'NEUTRAL' and base != htf:
        if abs(score) >= 6.0:
            notes.append(f"lower-timeframe evidence is overwhelming ({score:+.1f}), overriding the {htf} HTF bias")
        else:
            base = htf
            notes.append(f"HTF bias {htf} overrides conflicting lower-timeframe noise")
    firm = base if base else (htf if htf != 'NEUTRAL' and abs(score) >= 2.0 else None)
    now = datetime.now()
    stored = st.session_state.directional_bias.get(symbol)
    if stored and (now - stored['since']).total_seconds() < BIAS_MIN_HOLD_MINUTES * 60:
        standing = stored['direction']
        if firm and firm != standing:
            counter = picture['bull_items'] if firm == 'BUY' else picture['bear_items']
            has_htf_break = any(('H1' in it or 'H4' in it) and ('BOS' in it or 'CHOCH' in it) for it in counter)
            if has_htf_break and abs(score) >= 6.0:
                notes.append(f"standing {standing} bias overridden by an H1/H4 structural break with strong evidence ({score:+.1f})")
            elif abs(score) >= 4.0:
                firm = None
                notes.append(f"tape contradicts the standing {standing} bias but lacks an H1/H4 structural break - standing aside (WAIT) instead of flipping")
            else:
                firm = standing
                notes.append(f"maintaining the standing {standing} bias set at {stored['since'].strftime('%H:%M')} - insufficient proof to flip")
        elif firm is None and abs(score) < 4.0:
            firm = standing
            notes.append(f"tape is quiet - maintaining the standing {standing} bias set at {stored['since'].strftime('%H:%M')}")
        elif firm is None:
            notes.append(f"tape strongly contradicts the standing {standing} bias - standing aside (WAIT) until structure resolves")
    if firm:
        if not stored or stored['direction'] != firm:
            st.session_state.directional_bias[symbol] = {'direction': firm, 'since': now}
    elif stored and (now - stored['since']).total_seconds() >= BIAS_MIN_HOLD_MINUTES * 60:
        st.session_state.directional_bias.pop(symbol, None)
    return firm, notes

def compose_rich_reasoning(symbol, picture, firm, levels=None, dxy_context=None, extra_notes=None):
    snaps = {s['label']: s for s in picture.get('snaps', [])}
    def read(lbl):
        s = snaps.get(lbl)
        if not s:
            return 'neutral'
        m = s['micro']
        if m.get('momentum') == 'BULLISH' and m.get('price_vs_vwap') == 'ABOVE':
            return 'bullish'
        if m.get('momentum') == 'BEARISH' and m.get('price_vs_vwap') == 'BELOW':
            return 'bearish'
        return 'neutral'
    parts = []
    parts.append(f"Higher-timeframe structure reads {read('H4')} on H4 and {read('H1')} on H1 (HTF bias {picture.get('htf_bias', 'NEUTRAL')}); M30 {read('M30')}, M15 {read('M15')}, M10 {read('M10')}.")
    bull = picture.get('bull_items', [])
    bear = picture.get('bear_items', [])
    if bull:
        parts.append("Bullish evidence: " + "; ".join(bull[:4]) + ".")
    if bear:
        parts.append("Bearish evidence: " + "; ".join(bear[:4]) + ".")
    if firm:
        parts.append(f"Net directional verdict: {firm} with weighted evidence {picture.get('score', 0):+.1f}; the desk stands firmly on this bias until higher-timeframe structure breaks against it.")
    if dxy_context:
        t = str(dxy_context.get('trend', 'NEUTRAL')).lower()
        conf = 'confirming' if (firm == 'BUY' and t == 'bearish') or (firm == 'SELL' and t == 'bullish') else 'not confirming'
        parts.append(f"DXY is {t} with price {dxy_context.get('price_vs_vwap', 'NEUTRAL')} VWAP, {conf} the bias.")
    if levels:
        parts.append(f"Execution: entry {levels.get('entry')} at/near the live price, stop {levels.get('stop_loss')} beyond invalidation, target {levels.get('tp')} at the next liquidity zone ({levels.get('rr')}R).")
    if extra_notes:
        parts.extend(extra_notes)
    return ' '.join(parts)

def learning_note(symbol):
    stats = [s for s in st.session_state.learning_stats.get(symbol, []) if s != 'FLAT']
    if len(stats) < 3:
        return None, 0
    rate = sum(1 for s in stats if s == 'HIT') / len(stats)
    if rate >= 0.6:
        return f"Adaptive memory: recent hit-rate {rate * 100:.0f}% on {symbol}; conviction reinforced.", 3
    if rate <= 0.35:
        return f"Adaptive memory: recent hit-rate {rate * 100:.0f}% on {symbol}; requiring stronger evidence before trusting this bias.", -3
    return None, 0

def resolve_pending_outcomes(price_map):
    ledger = st.session_state.signal_ledger
    now = datetime.now()
    still = []
    for rec in ledger:
        age_min = (now - rec['time']).total_seconds() / 60.0
        px = price_map.get(rec['symbol'])
        if px is None or age_min < 20:
            still.append(rec)
            continue
        moved = (px - rec['entry']) if rec['direction'] == 'BUY' else (rec['entry'] - px)
        thr = rec['threshold']
        outcome = 'HIT' if moved >= thr else ('MISS' if moved <= -thr else 'FLAT')
        stats = st.session_state.learning_stats.setdefault(rec['symbol'], [])
        stats.append(outcome)
        st.session_state.learning_stats[rec['symbol']] = stats[-8:]
        if outcome != 'FLAT':
            add_notification('info', f"📚 Learning: {rec['symbol']} {rec['direction']} from {rec['time'].strftime('%H:%M')} resolved as {outcome} (price {px:.2f} vs entry {rec['entry']:.2f}).", symbol=rec['symbol'])
    st.session_state.signal_ledger = still

def cross_check_ai_evidence(analysis):
    ev = analysis.get('directional_evidence')
    if not isinstance(ev, dict):
        return analysis
    bull = ev.get('bullish') or []
    bear = ev.get('bearish') or []
    if not isinstance(bull, list) or not isinstance(bear, list):
        return analysis
    signal = analysis.get('signal')
    if signal == 'BUY' and len(bear) - len(bull) >= 2:
        analysis['confidence'] = 'MEDIUM' if analysis.get('confidence') == 'HIGH' else analysis.get('confidence')
        analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 74)
        analysis['reasoning'] = f"{analysis.get('reasoning', '')} Note: the AI's own evidence ledger was bearish-heavy, so bullish conviction was reduced."
    elif signal == 'SELL' and len(bull) - len(bear) >= 2:
        analysis['confidence'] = 'MEDIUM' if analysis.get('confidence') == 'HIGH' else analysis.get('confidence')
        analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 74)
        analysis['reasoning'] = f"{analysis.get('reasoning', '')} Note: the AI's own evidence ledger was bullish-heavy, so bearish conviction was reduced."
    return analysis

def apply_direction_correction_guard(analysis, confluence, symbol):
    signal = analysis.get('signal')
    if signal not in ('BUY', 'SELL') or not confluence:
        return analysis
    direction = confluence.get('direction')
    bull = confluence.get('bull_count', 0)
    bear = confluence.get('bear_count', 0)
    lead = abs(bull - bear)
    if direction and direction != signal and max(bull, bear) >= 3 and lead >= 2:
        analysis['signal'] = direction
        analysis['confidence'] = 'MEDIUM'
        analysis['confluence_score'] = max(MINIMUM_CONFLUENCE_SCORE, min(analysis.get('confluence_score', 0), 82))
        ev = confluence.get('bullish_evidence') if direction == 'BUY' else confluence.get('bearish_evidence')
        analysis['reasoning'] = f"{analysis.get('reasoning', '')} Direction corrected to {direction} by the structural evidence audit: {'; '.join(ev[:4])}."
        analysis['rejection_reason'] = None
    elif direction == signal:
        ev = confluence.get('bullish_evidence') if signal == 'BUY' else confluence.get('bearish_evidence')
        analysis['confluence_score'] = min(100, analysis.get('confluence_score', 0) + 2)
        analysis['reasoning'] = f"{analysis.get('reasoning', '')} Directional evidence audit confirms the {signal} side: {'; '.join(ev[:4])}."
    return analysis

def build_setup_context(df, swings, current_price, symbol, dxy_context=None, news_context=None):
    micro = calculate_microstructure(df)
    phase_context = detect_market_phase(df, swings=swings)
    continuation = detect_continuation(df)
    reversal = detect_reversal(df)
    exhaustion = detect_exhaustion(df)
    atr = calculate_atr(df)
    if continuation:
        setup_type = 'continuation'
        setup_bias = continuation.get('direction') or ('BUY' if micro.get('momentum') == 'BULLISH' else 'SELL')
    elif reversal:
        setup_type = 'reversal'
        setup_bias = reversal.get('direction') or ('BUY' if micro.get('momentum') == 'BULLISH' else 'SELL')
    elif exhaustion:
        setup_type = 'exhaustion'
        setup_bias = 'WAIT'
    else:
        setup_type = 'coiling'
        setup_bias = 'WAIT'
    entry_quality = phase_context.get('entry_quality', 'unknown')
    timing_state = 'ready' if entry_quality == 'early' else 'watch' if entry_quality == 'acceptable' else 'late'
    if setup_type == 'exhaustion' or entry_quality == 'late':
        timing_state = 'late'
    return {'setup_type': setup_type, 'setup_bias': setup_bias, 'phase': phase_context.get('phase', 'coiling'), 'phase_reason': phase_context.get('reason', 'Structure is forming.'), 'entry_quality': entry_quality, 'entry_timing': timing_state, 'atr': atr, 'micro': micro}

def build_execution_plan(df, current_price, signal, swings, pair_config=None, atr=None):
    pair_config = pair_config or get_pair_config('XAUUSD')
    atr_value = atr if atr is not None else calculate_atr(df)
    base_risk = current_price * pair_config.get('min_dist_pct', 0.0015)
    if atr_value is not None:
        base_risk = max(min(base_risk, 0.6 * atr_value), 0.4 * atr_value)
    base_risk = min(base_risk, current_price * pair_config.get('max_risk_pct', 0.008))
    if signal not in ['BUY', 'SELL']:
        return {'entry': round(current_price, 2), 'stop_loss': round(current_price, 2), 'take_profit': [round(current_price, 2)], 'risk_band': round(base_risk, 2), 'atr': round(atr_value, 2) if atr_value is not None else None}
    if signal == 'BUY':
        swing_lows = [level for level in swings.get('recent_swing_lows', []) if level < current_price]
        pivot = swing_lows[0] if swing_lows else None
        if pivot is not None:
            stop = min(current_price - max((current_price - pivot) * 0.9, base_risk), current_price - base_risk)
        else:
            stop = current_price - base_risk
        take_profit = current_price + (base_risk * pair_config.get('target_rr', 1.0))
    else:
        swing_highs = [level for level in swings.get('recent_swing_highs', []) if level > current_price]
        pivot = swing_highs[0] if swing_highs else None
        if pivot is not None:
            stop = max(current_price + max((pivot - current_price) * 0.9, base_risk), current_price + base_risk)
        else:
            stop = current_price + base_risk
        take_profit = current_price - (base_risk * pair_config.get('target_rr', 1.0))
    return {'entry': round(current_price, 2), 'stop_loss': round(stop, 2), 'take_profit': [round(take_profit, 2)], 'risk_band': round(base_risk, 2), 'atr': round(atr_value, 2) if atr_value is not None else None}

def update_market_state(new_state):
    if not new_state:
        return
    previous = st.session_state.get('market_state', 'coiling')
    if previous != new_state:
        st.session_state.state_history.append({'state': new_state, 'time': datetime.now().strftime('%H:%M:%S')})
        if len(st.session_state.state_history) > 20:
            st.session_state.state_history = st.session_state.state_history[-20:]
    st.session_state.market_state = new_state

# ── Signal Logic Helpers ─────────────────────────────────────────────────────
# FIX: hardened logic filter - only explicit self-invalidation phrases reject.
def validate_ai_logic(analysis):
    signal = analysis.get('signal')
    reasoning = analysis.get('reasoning', '').lower()
    micro_read = analysis.get('microstructure_read', '').lower()
    buy_bad = ['invalidates the buy', 'invalidates the long', 'invalid buy', 'invalid long',
               'buy setup is invalid', 'long setup is invalid', 'do not buy', "don't buy",
               'avoid buying', 'no buy setup', 'buy is invalidated']
    sell_bad = ['invalidates the sell', 'invalidates the short', 'invalid sell', 'invalid short',
                'sell setup is invalid', 'short setup is invalid', 'do not sell', "don't sell",
                'avoid selling', 'no sell setup', 'sell is invalidated']
    if signal == 'BUY':
        if any(p in reasoning for p in buy_bad):
            return False, 'The reasoning explicitly invalidates the BUY setup.'
        if any(t in micro_read for t in ['exhaustion', 'trap']):
            if not any(t in reasoning for t in ['pullback', 'retest', 'reclaim', 'confirmation', 'liquidity', 'sweep', 'zone', 'order block', 'fvg']):
                return False, 'Microstructure suggests a trap or exhaustion and the reasoning lacks a clear continuation or invalidation framework.'
    elif signal == 'SELL':
        if any(p in reasoning for p in sell_bad):
            return False, 'The reasoning explicitly invalidates the SELL setup.'
        if any(t in micro_read for t in ['exhaustion', 'trap']):
            if not any(t in reasoning for t in ['pullback', 'retest', 'reclaim', 'confirmation', 'liquidity', 'sweep', 'zone', 'order block', 'fvg']):
                return False, 'Microstructure suggests a trap or exhaustion and the reasoning lacks a clear continuation or invalidation framework.'
    return True, 'Valid'

def build_display_reason(analysis, symbol, current_price=None, phase_context=None, structural_context=None, dxy_context=None):
    reasoning = (analysis.get('reasoning') or '').strip()
    rejection = (analysis.get('rejection_reason') or '').strip()
    setup_context = analysis.get('setup_context') or {}
    phase = (phase_context or {}).get('phase') or setup_context.get('phase') or analysis.get('market_state') or 'unknown'
    setup_type = setup_context.get('setup_type') or analysis.get('market_state') or 'unknown'
    timing = setup_context.get('entry_timing') or (phase_context or {}).get('entry_quality') or 'unknown'
    signal = analysis.get('signal')
    score = analysis.get('confluence_score')
    confidence = analysis.get('confidence')
    dxy_status = analysis.get('dxy_correlation') or ('CONFIRMING' if dxy_context else '')
    entry = analysis.get('entry')
    current = current_price if current_price is not None else entry
    parts = []
    if reasoning:
        parts.append(reasoning)
    if analysis.get('confluence_breakdown'):
        parts.append(f"Confluence breakdown: {analysis.get('confluence_breakdown')}")
    else:
        breakdown_parts = []
        dxy = analysis.get('dxy_correlation') or 'N/A'
        micro = analysis.get('microstructure_read') or ''
        rsi_ctx = analysis.get('rsi_context') or ''
        struct_score = analysis.get('structural_score') if structural_context is None else structural_context.get('structural_score')
        live_price = analysis.get('live_price') or current_price
        if dxy:
            breakdown_parts.append(f"DXY: {dxy}")
        if micro:
            breakdown_parts.append(f"VWAP/RVOL: {micro}")
        if rsi_ctx:
            breakdown_parts.append(f"RSI: {rsi_ctx}")
        if struct_score is not None:
            breakdown_parts.append(f"Structure score: {struct_score}/100")
        if live_price is not None:
            try:
                breakdown_parts.append(f"Live price: {float(live_price):.2f}")
            except Exception:
                breakdown_parts.append(f"Live price: {live_price}")
        if breakdown_parts:
            parts.append('Confluence breakdown: ' + ' | '.join(breakdown_parts))
        else:
            parts.append(f"{symbol} is being assessed from the current market and execution context.")
    if phase and phase != 'unknown':
        parts.append(f"Market state is {phase}.")
    if setup_type and setup_type != 'unknown':
        parts.append(f"Setup type is {setup_type}.")
    if timing and timing != 'unknown':
        parts.append(f"Entry timing is {timing}.")
    if current is not None and entry is not None and current not in [0, None]:
        gap_pct = abs(entry - current) / current * 100 if current else 0.0
        parts.append(f"The proposed entry is about {gap_pct:.2f}% from the live price.")
    if dxy_status:
        parts.append(f"DXY correlation is {dxy_status.lower()}.")
    if score is not None:
        parts.append(f"Confluence score is {score}/100 with {confidence.lower() if confidence else 'unknown'} confidence.")
    if rejection and signal == 'WAIT':
        parts.append(f"Decision: {rejection}")
    elif rejection:
        parts.append(f"Decision: {rejection}")
    return ' '.join(parts)

def build_validation_detail(analysis, swings, current_price, symbol, pair_config=None, structural_context=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return 'No trade signal was produced because the setup did not meet the required structural or risk criteria.'
    if analysis.get('is_news_signal'):
        return 'Directional news signal accepted without strict entry/SL/TP enforcement.'
    pair_config = pair_config or {}
    min_dist = current_price * pair_config.get('min_dist_pct', 0.001)
    target_rr = pair_config.get('target_rr', 1.5)
    entry = analysis.get('entry', current_price)
    sl = analysis.get('stop_loss')
    tp_list = analysis.get('take_profit', [])
    tp1 = tp_list[0] if tp_list else None
    reasons = []
    if signal == 'BUY':
        if sl is None or sl >= entry:
            reasons.append(f'SL is not below entry ({sl} >= {entry}).')
        else:
            risk = entry - sl
            if risk + 1e-6 < min_dist:
                reasons.append(f'SL is too close to entry; risk is {risk:.4f}, below the minimum {min_dist:.4f} for {symbol}.')
        if tp1 is None or tp1 <= entry:
            reasons.append(f'TP is not above entry ({tp1} <= {entry}).')
        else:
            reward = tp1 - entry
            risk = entry - sl if sl is not None else 0
            if risk > 0 and (reward / risk) + 0.01 < target_rr:
                reasons.append(f'The proposed risk/reward is too low ({reward / risk:.2f} vs required {target_rr:.2f}).')
    else:
        if sl is None or sl <= entry:
            reasons.append(f'SL is not above entry ({sl} <= {entry}).')
        else:
            risk = sl - entry
            if risk + 1e-6 < min_dist:
                reasons.append(f'SL is too close to entry; risk is {risk:.4f}, below the minimum {min_dist:.4f} for {symbol}.')
        if tp1 is None or tp1 >= entry:
            reasons.append(f'TP is not below entry ({tp1} >= {entry}).')
        else:
            reward = entry - tp1
            risk = sl - entry if sl is not None else 0
            if risk > 0 and (reward / risk) + 0.01 < target_rr:
                reasons.append(f'The proposed risk/reward is too low ({reward / risk:.2f} vs required {target_rr:.2f}).')
    if structural_context and structural_context.get('structural_score', 0) < 70:
        reasons.append('The structural score is too weak for a high-quality setup.')
    return ' '.join(reasons) if reasons else 'The setup did not meet the structural and risk requirements for execution.'

def apply_conservative_signal_filter(analysis, structural_context, candles, dxy_context, news_context, current_price, swings, symbol, pair_config=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return analysis
    structural_score = (structural_context or {}).get('structural_score', 0)
    reasoning = (analysis.get('reasoning') or '').lower()
    recent_patterns = [c.get('pattern') for c in (candles or []) if c.get('pattern')]
    strong_recent = any(pattern in {'STRONG_BULLISH', 'STRONG_BEARISH', 'HAMMER', 'INVERTED_HAMMER', 'REJECTION_LOW', 'REJECTION_HIGH'} for pattern in recent_patterns)
    structural_markers = any(term in reasoning for term in ['order block', 'fvg', 'liquidity', 'retest', 'reclaim', 'zone', 'bos', 'choch', 'sweep'])
    if current_price is not None and swings:
        valid_swing_lows = [l for l in swings.get('recent_swing_lows', []) if l < current_price]
        valid_swing_highs = [h for h in swings.get('recent_swing_highs', []) if h > current_price]
    else:
        valid_swing_lows = []
        valid_swing_highs = []
    has_clear_anchor = bool((signal == 'BUY' and valid_swing_lows) or (signal == 'SELL' and valid_swing_highs))
    has_structure_support = structural_score >= 60 or strong_recent or structural_markers or (has_clear_anchor and structural_score >= 55)
    if not has_structure_support:
        analysis['confidence'] = 'LOW'
        analysis['confluence_score'] = max(analysis.get('confluence_score', 0), MINIMUM_CONFLUENCE_SCORE)
        analysis['rejection_reason'] = 'Structure is still forming, so the setup remains an early candidate rather than a hard no-trade.'
    return analysis

def validate_signal_math(analysis, pair_config=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return False, 'Invalid signal direction.'
    if analysis.get('is_news_signal'):
        return True, 'Direction-only news signal accepted.'

    pair_config = pair_config or {}
    entry = analysis.get('entry')
    sl = analysis.get('stop_loss')
    tp_list = analysis.get('take_profit', [])
    tp1 = tp_list[0] if tp_list else None

    current_price = entry if entry is not None else analysis.get('live_price') or analysis.get('price')
    if current_price is None:
        current_price = 0

    if entry is None or entry == 0:
        entry = current_price
        analysis['entry'] = round(float(entry), 2) if entry not in [None, ''] else 0

    if sl is None or tp1 is None:
        min_dist = float(pair_config.get('min_dist_pct', 0.0015)) * float(current_price or entry or 1)
        risk = max(min_dist, 0.5 if current_price else 0.01)
        target_rr = float(pair_config.get('target_rr', 1.5))
        if signal == 'BUY':
            sl = round(float(entry) - risk, 2)
            tp1 = round(float(entry) + (risk * target_rr), 2)
        else:
            sl = round(float(entry) + risk, 2)
            tp1 = round(float(entry) - (risk * target_rr), 2)
        analysis['stop_loss'] = sl
        analysis['take_profit'] = [tp1]
        analysis['levels_source'] = analysis.get('levels_source') or 'PYTHON'
        analysis['risk_band'] = round(abs(float(entry) - float(sl)), 2)

    entry = float(analysis.get('entry', entry) or 0)
    sl = float(analysis.get('stop_loss', sl) or 0)
    tp1 = float(analysis.get('take_profit', [tp1])[0] if analysis.get('take_profit') else tp1 or 0)

    if not entry or not sl or not tp1:
        return False, 'Missing entry, SL, or TP values.'

    if signal == 'BUY':
        if tp1 <= entry:
            return False, f'Invalid Math: For BUY, TP1 ({tp1}) MUST be > Entry ({entry}).'
        if sl >= entry:
            return False, f'Invalid Math: For BUY, SL ({sl}) MUST be < Entry ({entry}).'
    elif signal == 'SELL':
        if tp1 >= entry:
            return False, f'Invalid Math: For SELL, TP1 ({tp1}) MUST be < Entry ({entry}).'
        if sl <= entry:
            return False, f'Invalid Math: For SELL, SL ({sl}) MUST be > Entry ({entry}).'

    min_rr = pair_config.get('target_rr', 1.0)
    risk = abs(entry - sl)
    reward = abs(entry - tp1)
    if risk > 0 and (reward / risk) + 0.01 < min_rr:
        return False, f'Invalid Math: R:R is too low ({(reward / risk):.2f}). Minimum required is 1:{min_rr:.2f}.'
    return True, 'Valid'

def build_market_analysis_prompt():
    return """You are an elite institutional-style macro and execution analyst operating with the discipline of a professional trading desk. You have FULL access to the market conditions below: multi-timeframe structure (BOS/CHOCH, order blocks, FVGs, liquidity sweeps, swings), multi-timeframe RSI values and divergences, VWAP and RVOL microstructure, premium/discount range position, ATR volatility, DXY macro trend, higher-timeframe trend, and high-impact news. Use ALL of these concepts - miss nothing - and determine the current market state, the quality of the setup, and the correct directional edge yourself.
DATA PROVIDED:
{data_summary}
MICROSTRUCTURE (M10):
{microstructure_data}
STRUCTURE CONTEXT:
{structure_context}
MARKET STRUCTURE ZONES:
{market_structure_summary}
MULTI-TIMEFRAME CONTEXT (10M/15M/30M/1H/4H):
{multitimeframe_context}
RSI VALUES (MULTI-TIMEFRAME):
{rsi_values}
RSI / DIVERGENCE CONTEXT:
{rsi_context}
PREMIUM/DISCOUNT POSITION:
{premium_discount}
VOLATILITY (ATR):
{volatility_context}
HTF CONTEXT (H1/H4):
{htf_context}
DXY (US Dollar Index) TREND:
{dxy_data}
HISTORICAL CONTEXT:
{historical_context}
NEWS CONTEXT:
{news_summary}
STRUCTURAL SCORE (PYTHON):
{structural_score_context}
PYTHON DIRECTIONAL LEDGER (REFERENCE EVIDENCE):
{directional_ledger}
FIRM DESK BIAS (PYTHON, HTF-FIRST WITH HYSTERESIS):
{firm_bias}
MAX ENTRY DISTANCE FROM LIVE PRICE:
{max_entry_distance}
═══════════════════════════════════════════════════════════════════════════════
MANDATORY ANALYSIS RULES:
═══════════════════════════════════════════════════════════════════════════════
1. DETERMINE THE MARKET STATE YOURSELF (continuation, reversal, exhaustion, trend, coiling) using every concept provided.
2. THINK LIKE A PROFESSIONAL TRADER: weigh liquidity, flow, structure, volatility, relative volume, macro context, and execution quality together.
3. PRIORITIZE EARLY, PRICE-NEAR ENTRIES. Do not chase a large impulse.
4. ANALYZE RSI WITH INTENTION on every timeframe provided: overbought/oversold levels, regular and hidden divergences, confirmation vs contradiction.
5. USE DXY AS A CORE MACRO FILTER for XAUUSD, EURUSD, BTCUSD.
6. USE VWAP, RVOL, AND MICROSTRUCTURE AS EXECUTION INPUTS.
7. USE THE FULL STRUCTURE: BOS/CHOCH, order blocks, FVGs, liquidity sweeps, swing levels, support/resistance, candle behavior.
8. RESPECT PREMIUM/DISCOUNT: prefer buying in discount and selling in premium; treat trades in the opposite zone as lower quality.
9. IF A SETUP LOOKS LIKE EXHAUSTION OR A TRAP, do not force a trade.
10. SELECT AN ENTRY CLOSE TO THE LIVE PRICE.
11. PLACE SL BEYOND A CLEAR INVALIDATION POINT and TP AT THE NEXT MAJOR EXHAUSTION/LIQUIDITY ZONE with at least 1:1.5 R:R. SIZE THE STOP USING THE ATR CONTEXT.
12. YOUR REASONING MUST SHOW HOW THE CONFLUENCE WAS DERIVED from DXY, RSI, VWAP, RVOL, structure, premium/discount, volatility, and market phase.
13. WRITE PAIR-SPECIFIC, EXECUTION-FOCUSED, DETAILED REASONING (minimum 120 words). No generic filler.
14. ALWAYS TREAT THE LIVE PRICE AS THE PRIMARY REFERENCE.
15. DIRECTIONAL DECISION PROTOCOL (MANDATORY): (a) read the H4/H1 trend; (b) locate price in premium/discount; (c) list which liquidity side was swept; (d) list RSI divergences; (e) read the latest reversal/continuation candles; (f) weigh with this hierarchy: HTF trend > liquidity sweep + divergence > premium/discount > VWAP/momentum. Your final signal MUST equal the side that wins this hierarchy.
16. NEVER equate the prior impulse with the trade direction. A hard fall into a swept low that is now being rejected is a BUY reversal. A hard rally into a swept high that is being rejected is a SELL reversal.
17. FILL "directional_evidence" with separate bullish and bearish lists containing every item you relied on. Your "signal" MUST match the heavier list unless rule 15's hierarchy explicitly overrides it (explain any override in "reasoning").
18. BE DECISIVE AND STAND BY YOUR CALL: if one side wins the hierarchy or holds more evidence, output BUY or SELL. Output WAIT only when both evidence lists are empty or perfectly tied. Do not flip direction without new structural proof.
19. If DXY contradicts, it lowers confidence but does not by itself flip a direction decided by rule 15.
20. INTERNAL CONSISTENCY (MANDATORY): the numbers in "entry", "stop_loss", "take_profit" MUST be the exact levels you discuss in "reasoning" and "order_description".
21. ENTRY PROXIMITY (HARD RULE): your "entry" MUST be within the stated MAX ENTRY DISTANCE of the live price. If no valid setup can be placed within that distance, output WAIT.
OUTPUT JSON ONLY (NO MARKDOWN):
{{
    "market_state": "continuation|reversal|exhaustion|trend|coiling",
    "bias": "BULLISH|BEARISH|RANGING",
    "signal": "BUY|SELL|WAIT",
    "confluence_score": 0,
    "confidence": "HIGH|MEDIUM|LOW",
    "dxy_correlation": "CONFIRMING|CONTRADICTING|NEUTRAL",
    "microstructure_read": "Brief summary of VWAP/RVOL status and the intrabar read",
    "pre_news_bias": "If news < 2hrs: Expected trap and expansion direction. Else: N/A",
    "directional_evidence": {{"bullish": ["..."], "bearish": ["..."]}},
    "entry": 0.00,
    "stop_loss": 0.00,
    "take_profit": [0.00, 0.00],
    "rr_ratio": 0.00,
    "order_type": "MARKET|LIMIT|STOP|NONE",
    "order_description": "Brief execution plan using the SAME numbers as entry/stop_loss/take_profit.",
    "confluence_breakdown": "A short explanation of the weighting behind the score using DXY, RSI, VWAP, RVOL, structure, premium/discount, and market phase.",
    "reasoning": "A detailed, confident, pair-specific institutional brief (min 120 words) showing HTF structure, manipulation reads, divergences, DXY, volatility, and exact invalidation/target logic.",
    "rejection_reason": "If WAIT, explain the exact missing condition in a trader-friendly way."
}}
"""

# FIX: News prompt now asks for direction-only (no Entry/SL/TP) and asks AI to use historical patterns + consensus expectations.
def build_news_analysis_prompt():
    return """You are an elite news-driven macro analyst operating with the discipline of a professional trading desk. You analyze high-impact USD-sensitive news BEFORE it is released using a SYSTEMATIC MULTI-LAYER ANALYSIS to project how the event will affect trading pairs AT THE TIME of the news reading.

═══════════════════════════════════════════════════════════════════════════════
SYSTEMATIC ANALYSIS FRAMEWORK (FOLLOW IN ORDER):
═══════════════════════════════════════════════════════════════════════════════

LAYER 1 - HISTORICAL RELEASE PATTERN ANALYSIS:
- Recall the LAST 3-4 RELEASES of this specific event type
- For each release: What was the previous value? What was the consensus? What was the actual? What was the surprise?
- How did the market react to each surprise? (direction, magnitude, duration)
- What is the typical consensus expectation for THIS release?
- What would constitute a surprise vs consensus for THIS release?

LAYER 2 - CURRENT MARKET POSITIONING ANALYSIS:
- Analyze the CURRENT MARKET STRUCTURE provided (structure context, HTF context)
- Where is price positioned relative to key levels? (premium/discount, key support/resistance)
- What is the current momentum and trend across timeframes?
- What is the current RSI positioning across timeframes?
- What is the current DXY trend and positioning?
- Based on current positioning, is the market positioned FOR or AGAINST the expected news outcome?

LAYER 3 - NEWS IMPACT MECHANICS:
- How does THIS specific event type typically affect the US Dollar?
- How does THIS specific event type typically affect EACH SYMBOL (XAUUSD, EURUSD, BTCUSD, US30)?
- What is the typical reaction pattern? (immediate spike, delayed reaction, fade, continuation)
- What time of day is the release? (affects liquidity and reaction magnitude)
- What is the current market session? (affects liquidity and reaction magnitude)

LAYER 4 - CROSS-ASSET CORRELATION ANALYSIS:
- How do different symbols typically react to THIS event type?
- Are there any cross-asset correlations that confirm or contradict the expected move?
- Are there any divergences between assets that suggest a specific outcome?

LAYER 5 - SYNTHESIS AND DIRECTIONAL EDGE:
- Combine all layers to determine the EXPECTED NEWS OUTCOME (stronger/weaker USD)
- Determine the EXPECTED SYMBOL REACTION for each symbol
- Determine if current positioning is FOR or AGAINST the expected outcome
- Determine the directional edge: Should the trader be positioned LONG or SHORT when the news drops?

DATA PROVIDED:
{data_summary}
MICROSTRUCTURE (M10):
{microstructure_data}
STRUCTURE CONTEXT:
{structure_context}
RSI VALUES (MULTI-TIMEFRAME):
{rsi_values}
PREMIUM/DISCOUNT POSITION:
{premium_discount}
VOLATILITY (ATR):
{volatility_context}
HTF CONTEXT (H1/H4):
{htf_context}
DXY (US Dollar Index) TREND:
{dxy_data}
HISTORICAL CONTEXT:
{historical_context}
NEWS EVENT DETAILS:
{news_summary}
STRUCTURAL SCORE (PYTHON):
{structural_score_context}
PYTHON DIRECTIONAL LEDGER (REFERENCE EVIDENCE):
{directional_ledger}

═══════════════════════════════════════════════════════════════════════════════
MANDATORY ANALYSIS RULES:
═══════════════════════════════════════════════════════════════════════════════

1. COMPLETE ALL 5 LAYERS OF ANALYSIS before determining the final signal
2. For EACH SYMBOL, determine:
   - Expected USD impact (stronger/weaker/neutral)
   - Expected symbol reaction (up/down/neutral)
   - Current positioning (for/against the expected move)
   - Final directional edge (long/short/neutral)
3. Be SPECIFIC about historical patterns - cite specific previous releases and reactions
4. Be SPECIFIC about current positioning - cite specific levels and indicators
5. Be SPECIFIC about the expected reaction - cite the mechanism and timing
6. Output BUY or SELL when there is a clear directional edge
7. NEVER output WAIT for a news signal. Always pick BUY or SELL with detailed reasoning.
8. DO NOT output Entry, SL, or TP levels - focus ONLY on direction and reasoning
9. Write DETAILED reasoning (minimum 200 words) that shows your complete analysis process

OUTPUT JSON ONLY (NO MARKDOWN):
{{
    "market_state": "continuation|reversal|exhaustion|trend|coiling",
    "bias": "BULLISH|BEARISH|RANGING",
    "signal": "BUY|SELL",
    "confluence_score": 0,
    "confidence": "HIGH|MEDIUM|LOW",
    "dxy_correlation": "CONFIRMING|CONTRADICTING|NEUTRAL",
    "microstructure_read": "Brief summary of VWAP/RVOL status",
    "pre_news_bias": "Detailed explanation of expected USD impact and symbol reaction",
    "directional_evidence": {{"bullish": ["..."], "bearish": ["..."]}},
    "historical_pattern": "Detailed analysis of last 3-4 releases: previous values, consensus, actual, surprises, and market reactions",
    "current_positioning": "Detailed analysis of current market positioning relative to expected news outcome",
    "news_impact_mechanics": "Detailed explanation of how this event type affects USD and each symbol, including typical reaction patterns and timing",
    "reasoning": "Complete synthesis of all 5 layers showing your complete analysis process (minimum 200 words)",
    "rejection_reason": "If WAIT, detailed explanation of why there is no clear directional edge"
}}
"""

MINIMUM_CONFLUENCE_SCORE = 70
ANALYSIS_INTERVAL_MINUTES = 5
NEWS_PRE_WINDOW_HOURS = 2

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Der-AI | Quantitative Macro System", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

def render_countdown_timer():
    if st.session_state.bot_running and st.session_state.next_check_time:
        scheduled_iso = st.session_state.next_check_time.isoformat()
        components.html("""
            <script>
            const scheduledAt = new Date('""" + scheduled_iso + """');
            if (!window.__deraiCountdown) {
                window.__deraiCountdown = setInterval(function() {
                    const now = new Date();
                    const deltaSeconds = Math.max(0, Math.round((scheduledAt - now) / 1000));
                    const countdownEls = [
                        window.parent.document.getElementById('derai-countdown-text'),
                        window.parent.document.getElementById('derai-sidebar-countdown')
                    ];
                    const minutes = Math.floor(deltaSeconds / 60);
                    const seconds = deltaSeconds % 60;
                    const countdownText = minutes + 'm ' + String(seconds).padStart(2, '0') + 's';
                    countdownEls.forEach(el => { if (el) { el.innerText = countdownText; } });
                }, 1000);
            }
            </script>
            """, height=0)

# ── Initialize Session State ──────────────────────────────────────────────────
if 'bot_running' not in st.session_state: st.session_state.bot_running = False
if 'last_analysis_time' not in st.session_state: st.session_state.last_analysis_time = None
if 'signal_history' not in st.session_state: st.session_state.signal_history = []
if 'next_check_time' not in st.session_state: st.session_state.next_check_time = None
if 'active_signals' not in st.session_state: st.session_state.active_signals = {}
if 'notifications' not in st.session_state: st.session_state.notifications = []
if 'unread_count' not in st.session_state: st.session_state.unread_count = 0
if 'rate_limit_hit' not in st.session_state: st.session_state.rate_limit_hit = False
if 'cached_market_data' not in st.session_state: st.session_state.cached_market_data = {}
if 'last_market_fetch_time' not in st.session_state: st.session_state.last_market_fetch_time = None
if 'cached_analysis' not in st.session_state: st.session_state.cached_analysis = {}
if 'gpt_rate_limit_until' not in st.session_state: st.session_state.gpt_rate_limit_until = None
if 'gpt_rate_limit_reason' not in st.session_state: st.session_state.gpt_rate_limit_reason = ''
if 'last_gpt_request_time' not in st.session_state: st.session_state.last_gpt_request_time = None
if 'gpt_tokens_used' not in st.session_state: st.session_state.gpt_tokens_used = 0
if 'gpt_token_window_start' not in st.session_state: st.session_state.gpt_token_window_start = datetime.now()
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = {'current_pair': None, 'message': None, 'estimated_tokens': None, 'estimated_duration': None}
if 'news_event_statuses' not in st.session_state: st.session_state.news_event_statuses = {}
if 'active_news_event' not in st.session_state: st.session_state.active_news_event = None
if 'news_signal_sent' not in st.session_state: st.session_state.news_signal_sent = {}
if 'news_event_results' not in st.session_state: st.session_state.news_event_results = {}
if 'market_state' not in st.session_state: st.session_state.market_state = 'coiling'
if 'state_history' not in st.session_state: st.session_state.state_history = []
if 'analysis_in_progress' not in st.session_state: st.session_state.analysis_in_progress = False
if 'directional_bias' not in st.session_state: st.session_state.directional_bias = {}
if 'signal_ledger' not in st.session_state: st.session_state.signal_ledger = []
if 'learning_stats' not in st.session_state: st.session_state.learning_stats = {}

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

try:
    query_params = {k: v[0] for k, v in st.query_params.to_dict().items()}
except Exception:
    query_params = {}

TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "")
SYMBOLS = ['XAUUSD', 'EURUSD', 'BTCUSD', 'US30']
YFINANCE_MAP = {'XAUUSD': 'GC=F', 'EURUSD': 'EURUSD=X', 'BTCUSD': 'BTC-USD', 'US30': '^DJI', 'DXY': 'DX-Y.NYB'}

# ── Notification System ───────────────────────────────────────────────────────
def add_notification(note_type, message, symbol=None, signal=None, score=None):
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    notification = {'id': str(uuid.uuid4()), 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': note_type, 'message': message, 'symbol': symbol, 'signal': signal, 'score': score, 'read': False}
    st.session_state.notifications.append(notification)
    if len(st.session_state.notifications) > 200:
        st.session_state.notifications = st.session_state.notifications[-200:]
    st.session_state.unread_count = sum(1 for n in st.session_state.notifications if not n.get('read', False))
    return notification

def get_notifications():
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    return st.session_state.notifications

def mark_all_notifications_read():
    if 'notifications' in st.session_state:
        for n in st.session_state.notifications:
            n['read'] = True
        st.session_state.unread_count = 0

def clear_notifications():
    st.session_state.notifications = []
    st.session_state.unread_count = 0

def debug_log(message, exc_info=False):
    return

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return False
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def build_telegram_signal_message(symbol, result):
    tp_values = result.get('take_profit', [])
    tp_value = tp_values[0] if tp_values else 'N/A'
    score = result.get('confluence_score', 0)
    event_time = result.get('news_time') or 'now'
    return (f"🌍 <b>DER-AI MACRO SIGNAL</b>\n"
            f"📊 <b>{symbol}</b> - {result.get('signal')}\n"
            f"🗂️ Event: Live market Analysis\n"
            f"⏰ Time: {event_time}\n"
            f"📈 Score: {score}/100\n"
            f"💰 Entry: {result.get('entry')} | 🛑 SL: {result.get('stop_loss')} | 🎯 TP: {tp_value}\n"
            f"📈 DXY: {result.get('dxy_correlation')}\n"
            f"🧠 {result.get('reasoning')}")

# ── Pair-Specific Configuration & Structural Scoring ─────────────────────────
# FIX: cooldown 10 -> 30 so a good setup is not re-signalled every cycle.
def get_pair_config(symbol):
    configs = {
        'XAUUSD': {'min_dist_pct': 0.0015, 'max_risk_pct': 0.008, 'target_rr': 1.0, 'score_floor': MINIMUM_CONFLUENCE_SCORE, 'candidate_score': MINIMUM_CONFLUENCE_SCORE, 'cooldown_minutes': 30, 'max_entry_gap_pct': 0.003, 'max_entry_points': 10},
        'EURUSD': {'min_dist_pct': 0.0005, 'max_risk_pct': 0.003, 'target_rr': 1.0, 'score_floor': MINIMUM_CONFLUENCE_SCORE, 'candidate_score': MINIMUM_CONFLUENCE_SCORE, 'cooldown_minutes': 30, 'max_entry_gap_pct': 0.002, 'max_entry_points': 0.0025},
        'BTCUSD': {'min_dist_pct': 0.004, 'max_risk_pct': 0.012, 'target_rr': 1.0, 'score_floor': MINIMUM_CONFLUENCE_SCORE, 'candidate_score': MINIMUM_CONFLUENCE_SCORE, 'cooldown_minutes': 30, 'max_entry_gap_pct': 0.004, 'max_entry_points': 150},
        'US30': {'min_dist_pct': 0.003, 'max_risk_pct': 0.008, 'target_rr': 1.0, 'score_floor': MINIMUM_CONFLUENCE_SCORE, 'candidate_score': MINIMUM_CONFLUENCE_SCORE, 'cooldown_minutes': 30, 'max_entry_gap_pct': 0.003, 'max_entry_points': 120},
    }
    return configs.get(symbol, {'min_dist_pct': 0.001, 'max_risk_pct': 0.008, 'target_rr': 1.0, 'score_floor': MINIMUM_CONFLUENCE_SCORE, 'candidate_score': MINIMUM_CONFLUENCE_SCORE, 'cooldown_minutes': 30, 'max_entry_gap_pct': 0.003, 'max_entry_points': 10})

def calculate_structural_score(df, symbol, dxy_context=None, news_context=None, phase_context=None):
    if df.empty or len(df) < 10:
        return {'structural_score': 0, 'score_reason': 'Insufficient data', 'candidate_direction': None, 'market_phase': 'coiling', 'phase_reason': 'Not enough data to assess structure.'}
    micro = calculate_microstructure(df)
    candles = analyze_candle_structure(df)
    bos, choch = detect_bos_choch(df)
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    sweeps = detect_liquidity_sweeps(df)
    phase_context = phase_context or detect_market_phase(df)
    score = 42
    reasons = []
    if micro.get('price_vs_vwap') == 'ABOVE':
        score += 8; reasons.append('price holding above VWAP')
    else:
        score += 4; reasons.append('price trading near VWAP')
    if micro.get('momentum') == 'BULLISH':
        score += 6; reasons.append('short-term momentum bullish')
    else:
        score += 4; reasons.append('short-term momentum bearish')
    if micro.get('rvol', 0) > 2.0:
        score += 10; reasons.append('strong institutional volume')
    elif micro.get('rvol', 0) < 0.5:
        score -= 8; reasons.append('low volume / exhaustion risk')
    if bos == 'BULLISH_BOS' or choch == 'BULLISH_CHOCH':
        score += 10; reasons.append('bullish BOS/CHOCH')
    elif bos == 'BEARISH_BOS' or choch == 'BEARISH_CHOCH':
        score += 10; reasons.append('bearish BOS/CHOCH')
    if order_blocks:
        score += 6; reasons.append('order block present')
    if fvgs:
        score += 6; reasons.append('fair value gap present')
    if sweeps:
        score += 6; reasons.append('liquidity sweep detected')
    phase = phase_context.get('phase')
    if phase == 'continuation':
        score += 8; reasons.append('continuation structure is present')
    elif phase == 'reversal':
        score += 6; reasons.append('reversal structure is forming')
    elif phase == 'exhaustion':
        score -= 5; reasons.append('exhaustion is present and needs caution')
    entry_quality = phase_context.get('entry_quality')
    if entry_quality == 'early':
        score += 5; reasons.append('entry zone is still early and actionable')
    elif entry_quality == 'late':
        score -= 4; reasons.append('entry zone is late and may be chasing price')
    if dxy_context and symbol in ['XAUUSD', 'EURUSD', 'BTCUSD']:
        if dxy_context['trend'] == 'BULLISH' and dxy_context['price_vs_vwap'] == 'ABOVE':
            score -= 6; reasons.append('DXY is suppressing the setup')
        elif dxy_context['trend'] == 'BEARISH' and dxy_context['price_vs_vwap'] == 'BELOW':
            score += 6; reasons.append('DXY is supporting the setup')
    if news_context and news_context.get('within_2h'):
        if news_context.get('bias') == 'opposite':
            score -= 4; reasons.append('news risk reduces conviction')
        else:
            score += 2; reasons.append('news context remains supportive')
    score = max(0, min(100, int(score)))
    phase_dir = phase_context.get('direction')
    candidate_direction = None
    if phase_dir and score >= 70:
        candidate_direction = phase_dir
    elif score >= 75 and micro.get('momentum') == 'BULLISH':
        candidate_direction = 'BUY'
    elif score >= 75 and micro.get('momentum') == 'BEARISH':
        candidate_direction = 'SELL'
    return {'structural_score': score, 'score_reason': '; '.join(reasons[-4:]), 'candidate_direction': candidate_direction, 'market_phase': phase, 'phase_reason': phase_context.get('reason', 'Structure is being assessed.'), 'entry_quality': entry_quality, 'entry_zone': phase_context.get('entry_zone')}

# ── SMC & Microstructure Engines ─────────────────────────────────────────────
def analyze_candle_structure(df):
    if len(df) < 3: return []
    analysis = []
    for i in range(max(0, len(df)-10), len(df)):
        candle = df.iloc[i]
        body = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        if total_range == 0: continue
        upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        body_ratio = body / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        candle_type = "BULLISH" if candle['Close'] > candle['Open'] else "BEARISH"
        pattern = "NORMAL"
        if body_ratio > 0.7: pattern = "STRONG_" + candle_type
        elif body_ratio < 0.3: pattern = "DOJI"
        elif upper_wick_ratio > 0.6: pattern = "REJECTION_HIGH"
        elif lower_wick_ratio > 0.6: pattern = "REJECTION_LOW"
        elif upper_wick_ratio > 0.4 and body_ratio < 0.4: pattern = "SHOOTING_STAR" if candle_type == "BEARISH" else "HANGING_MAN"
        elif lower_wick_ratio > 0.4 and body_ratio < 0.4: pattern = "HAMMER" if candle_type == "BULLISH" else "INVERTED_HAMMER"
        analysis.append({'time': df.index[i], 'candle_type': candle_type, 'pattern': pattern, 'body_ratio': body_ratio, 'upper_wick_ratio': upper_wick_ratio, 'lower_wick_ratio': lower_wick_ratio, 'price': candle['Close'], 'volume': candle['Volume']})
    return analysis[-5:]

def calculate_microstructure(df):
    if df is None or len(df) < 2:
        return {}
    try:
        typical_price = (pd.to_numeric(df['High'], errors='coerce') + pd.to_numeric(df['Low'], errors='coerce') + pd.to_numeric(df['Close'], errors='coerce')) / 3
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
        current_vwap = float(vwap.iloc[-1])
        current_price = float(pd.to_numeric(df['Close'], errors='coerce').iloc[-1])
        avg_volume = volume.rolling(window=min(20, len(volume))).mean().iloc[-1]
        current_volume = float(volume.iloc[-1])
        rvol = current_volume / avg_volume if avg_volume > 0 else 1.0
        anchor_index = -min(5, len(df))
        price_change = current_price - float(pd.to_numeric(df['Close'], errors='coerce').iloc[anchor_index])
        return {"vwap": round(current_vwap, 2), "price_vs_vwap": "ABOVE" if current_price > current_vwap else "BELOW", "rvol": round(rvol, 2), "volume_anomaly": "HIGH_INSTITUTIONAL" if rvol > 2.0 else "NORMAL", "momentum": "BULLISH" if price_change > 0 else "BEARISH"}
    except Exception:
        return {}

def detect_bos_choch(df):
    if df is None or len(df) < 2:
        return None, None
    try:
        highs = pd.to_numeric(df['High'], errors='coerce').dropna()
        lows = pd.to_numeric(df['Low'], errors='coerce').dropna()
        if highs.empty or lows.empty or len(df) < 4:
            return None, None
        recent_high = float(highs.iloc[-1])
        prev_high = float(highs.iloc[-2]) if len(highs) >= 2 else recent_high
        recent_low = float(lows.iloc[-1])
        prev_low = float(lows.iloc[-2]) if len(lows) >= 2 else recent_low
        bos, choch = None, None
        if recent_high > prev_high * 1.001:
            bos = "BULLISH_BOS"
        elif recent_low < prev_low * 0.999:
            bos = "BEARISH_BOS"
        if bos == "BULLISH_BOS" and recent_low > prev_low:
            choch = "BULLISH_CHOCH"
        elif bos == "BEARISH_BOS" and recent_high < prev_high:
            choch = "BEARISH_CHOCH"
        return bos, choch
    except Exception:
        return None, None

def find_swings(df, window=5):
    highs = df['High'].rolling(window * 2 + 1, center=True).max()
    lows = df['Low'].rolling(window * 2 + 1, center=True).min()
    return {"recent_swing_highs": [round(p, 5) for p in df['High'][df['High'] == highs].tail(4).tolist()], "recent_swing_lows": [round(p, 5) for p in df['Low'][df['Low'] == lows].tail(4).tolist()]}

def detect_order_blocks(df):
    if len(df) < 5: return []
    order_blocks = []
    for i in range(len(df)-3, len(df)):
        if i < 2: continue
        candle, prev_candle = df.iloc[i], df.iloc[i-1]
        if (candle['Close'] > candle['Open'] and (candle['Close'] - candle['Open']) > (candle['High'] - candle['Low']) * 0.6 and prev_candle['Close'] < prev_candle['Open']):
            order_blocks.append({'type': 'BULLISH_OB', 'price': candle['Low'], 'strength': 'STRONG' if (candle['Close'] - candle['Open']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'})
        if (candle['Close'] < candle['Open'] and (candle['Open'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.6 and prev_candle['Close'] > prev_candle['Open']):
            order_blocks.append({'type': 'BEARISH_OB', 'price': candle['High'], 'strength': 'STRONG' if (candle['Open'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'})
    return order_blocks[-3:]

def detect_fvg(df):
    if len(df) < 3: return []
    fvgs = []
    for i in range(len(df)-2, len(df)):
        if i < 2: continue
        curr, prev, prev2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        if prev['Low'] > prev2['High'] and curr['Low'] > prev['High']:
            fvgs.append({'type': 'BULLISH_FVG', 'top': prev['Low'], 'bottom': prev2['High']})
        if prev['High'] < prev2['Low'] and curr['High'] < prev['Low']:
            fvgs.append({'type': 'BEARISH_FVG', 'top': prev2['Low'], 'bottom': prev['High']})
    return fvgs[-2:]

def detect_liquidity_sweeps(df):
    if len(df) < 10: return []
    sweeps = []
    recent = df.tail(10)
    for i in range(1, len(recent)):
        candle, prev = recent.iloc[i], recent.iloc[i-1]
        if (candle['Low'] < prev['Low'] * 0.999 and candle['Close'] > candle['Open'] and (candle['Close'] - candle['Low']) > (candle['High'] - candle['Low']) * 0.6):
            sweeps.append({'type': 'BULLISH_SWEEP', 'price': candle['Low'], 'strength': 'STRONG' if (candle['Close'] - candle['Low']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'})
        if (candle['High'] > prev['High'] * 1.001 and candle['Close'] < candle['Open'] and (candle['High'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.6):
            sweeps.append({'type': 'BEARISH_SWEEP', 'price': candle['High'], 'strength': 'STRONG' if (candle['High'] - candle['Close']) > (candle['High'] - candle['Low']) * 0.8 else 'MODERATE'})
    return sweeps[-2:]

# ── Data Fetching ─────────────────────────────────────────────────────────────
def _build_dataframe_from_records(records):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    if 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    elif 'datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['datetime'], utc=True)
    else:
        return pd.DataFrame()
    df = df.set_index('Date').sort_index()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

def fetch_candles_from_bitfinex(symbol, interval, limit=200):
    pair_map = {'BTCUSD': 'tBTCUSD', 'XAUUSD': 'tXAUUSD', 'EURUSD': 'tEURUSD', 'DXY': None}
    bitfinex_symbol = pair_map.get(symbol)
    if not bitfinex_symbol:
        return pd.DataFrame()
    interval_map = {'15m': '15m', '30m': '30m', '60m': '1h', '1h': '1h', '4h': '4h'}
    interval_code = interval_map.get(interval)
    if not interval_code:
        return pd.DataFrame()
    try:
        url = f'https://api-pub.bitfinex.com/v2/candles/trade:{interval_code}:{bitfinex_symbol}/hist?limit={limit}'
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return pd.DataFrame()
        records = []
        for item in payload:
            if not isinstance(item, list) or len(item) < 6:
                continue
            ts, open_price, close_price, high_price, low_price, volume = item[:6]
            records.append({'timestamp': ts, 'Open': float(open_price), 'High': float(high_price), 'Low': float(low_price), 'Close': float(close_price), 'Volume': float(volume)})
        return _build_dataframe_from_records(records)
    except Exception as exc:
        print(f"⚠️ Bitfinex fetch failed for {symbol} [{interval}]: {exc}")
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_ohlcv(yf_symbol, interval, period):
    if yf is None:
        return pd.DataFrame()
    for symbol in ['BTCUSD', 'XAUUSD', 'EURUSD', 'DXY']:
        if yf_symbol in {symbol, YFINANCE_MAP.get(symbol, symbol)}:
            direct_df = fetch_candles_from_bitfinex(symbol, interval, limit=250)
            if not direct_df.empty:
                return direct_df
            break
    try:
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False, timeout=30)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        if df.empty:
            return pd.DataFrame()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        if 'Date' in df.columns:
            df = df.set_index('Date')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        return df if len(df) >= 5 else pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Yahoo fetch failed for {yf_symbol} [{interval}/{period}]: {e}")
        return pd.DataFrame()

def fetch_symbol_data(symbol, yf_symbol):
    df_m10 = pd.DataFrame()
    for interval, period in [('10m', '5d'), ('15m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m10 = df_candidate
            break
    if df_m10.empty:
        for interval, period in [('15m', '5d'), ('30m', '5d'), ('60m', '5d')]:
            df_candidate = fetch_ohlcv(yf_symbol, interval, period)
            if not df_candidate.empty:
                df_m10 = df_candidate
                break
    df_m15 = pd.DataFrame()
    for interval, period in [('15m', '5d'), ('30m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m15 = df_candidate
            break
    df_m30 = pd.DataFrame()
    for interval, period in [('30m', '5d'), ('60m', '5d')]:
        df_candidate = fetch_ohlcv(yf_symbol, interval, period)
        if not df_candidate.empty:
            df_m30 = df_candidate
            break
    df_h1 = fetch_ohlcv(yf_symbol, '1h', '1mo')
    if df_h1.empty:
        df_h1 = fetch_ohlcv(yf_symbol, '60m', '1mo')
    df_h4 = fetch_ohlcv(yf_symbol, '4h', '3mo')
    if df_h4.empty:
        df_h4 = fetch_ohlcv(yf_symbol, '1d', '6mo')
    if df_m10.empty and not df_m15.empty:
        df_m10 = df_m15
    if df_m30.empty and not df_h1.empty:
        df_m30 = df_h1
    return {'M10': df_m10, 'M15': df_m15, 'M30': df_m30, 'H1': df_h1, 'H4': df_h4}

@st.cache_data(ttl=120, show_spinner=False)
def fetch_all_data():
    data = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=min(5, len(YFINANCE_MAP))) as executor:
        for symbol, yf_symbol in YFINANCE_MAP.items():
            futures[executor.submit(fetch_symbol_data, symbol, yf_symbol)] = symbol
    for future in as_completed(futures):
        symbol = futures[future]
        try:
            data[symbol] = future.result()
        except Exception as e:
            print(f"❌ Exception fetching {symbol}: {str(e)}")
            data[symbol] = {'M10': pd.DataFrame(), 'H1': pd.DataFrame(), 'H4': pd.DataFrame()}
    return data

# FIX: Allow MEDIUM impact events if USD-sensitive (e.g., Jobless Claims)
def get_high_impact_news(selected_symbols=None, reference_dt=None):
    endpoints = [
        "https://r.jina.ai/http://https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://r.jina.ai/http://https://nfs.faireconomy.media/ff_calendar_thisweek.json?apifooter=false",
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json?apifooter=false",
    ]
    for url in endpoints:
        try:
            res = requests.get(url, params={"apifooter": "false"}, timeout=20)
            res.raise_for_status()
            text = res.text
            if text.startswith("Title:") or "Markdown Content:" in text:
                text = text.split("Markdown Content:", 1)[-1].strip()
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get('events') or payload.get('items') or payload.get('data') or []
            if not isinstance(payload, list):
                continue
            reference_dt = reference_dt or datetime.now(timezone.utc)
            # FIX: only_high=False to include MEDIUM USD-sensitive events
            events = parse_news_payload(payload, reference_dt=reference_dt, lookahead_hours=168, only_high=False)
            if events:
                filtered_events = filter_relevant_news([{'event': e['event'], 'currency': e['currency'], 'impact': e['impact'], 'time': e['time'], 'event_time_utc': e['event_time_utc'], 'minutes_until': e['minutes_until'], 'within_2h': e['within_2h'], 'timezone': e['timezone']} for e in events], selected_symbols=selected_symbols, reference_dt=reference_dt)
                return [{'time': format_east_africa_time(e['event_time_utc']), 'currency': e['currency'], 'event': e['event'], 'impact': e['impact'], 'minutes_until': e['minutes_until'], 'within_2h': e['within_2h'], 'timezone': e['timezone'], 'event_time_utc': e['event_time_utc'], 'event_id': f"{e['event']}|{e['currency']}|{e['event_time_utc'].strftime('%Y-%m-%d %H:%M:%S')}"} for e in filtered_events]
        except Exception as exc:
            print(f"⚠️ News fetch failed for {url}: {exc}")
    return []

def sync_news_event_statuses(news_events, selected_symbols=None):
    statuses = st.session_state.news_event_statuses
    for event in news_events:
        event_id = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
        if not statuses.get(event_id):
            statuses[event_id] = {'event': event.get('event'), 'currency': event.get('currency'), 'time': event.get('time'), 'status': 'waiting', 'detail': 'Waiting for AI pre-news analysis (sent once, >=2h before release).'}
    stale_ids = [k for k in statuses if not any(e.get('event_id') == k for e in news_events)]
    for stale_id in stale_ids:
        del statuses[stale_id]
        st.session_state.news_signal_sent.pop(stale_id, None)
    st.session_state.news_event_statuses = statuses

def update_news_event_status(event, status, detail=None):
    if not event:
        return
    event_id = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
    st.session_state.news_event_statuses[event_id] = {'event': event.get('event'), 'currency': event.get('currency'), 'time': event.get('time'), 'status': status, 'detail': detail or ''}

# ── Dedicated Pre-News Impact Engine (sent ONCE, >=2h ahead, Calendar+Telegram only) ──
def wait_for_groq_spacing():
    if st.session_state.last_gpt_request_time:
        delta = (datetime.now() - st.session_state.last_gpt_request_time).total_seconds()
        if delta < GROQ_MIN_REQUEST_INTERVAL:
            time.sleep(GROQ_MIN_REQUEST_INTERVAL - delta)

def pick_news_event_for_analysis(news_events, now):
    best = None

    for event in news_events or []:
        et = event.get('event_time_utc')

        if not et:
            continue

        eid = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"
        status = st.session_state.news_event_statuses.get(eid, {}).get('status', 'waiting')

        # HARD TOKEN GUARD:
        # If results already exist, or the event was already marked sent, skip it forever.
        if st.session_state.news_event_results.get(eid) or st.session_state.news_signal_sent.get(eid):
            if status not in ('sent', 'window_closed', 'expired'):
                update_news_event_status(
                    event,
                    'sent',
                    'Already analyzed; skipping repeat AI run to save tokens.'
                )
            continue

        # Terminal states should never be analyzed again.
        if status in ('sent', 'window_closed', 'expired'):
            continue

        # Event already passed.
        if et <= now:
            update_news_event_status(
                event,
                'expired',
                'Event time passed before analysis.'
            )
            continue

        lead_time = et - now

        # OPTIONAL EXTRA TOKEN SAVER:
        # If you only want AI to analyze events when they are inside the 2-hour pre-news window,
        # remove the # from the next two lines.
        #
        # if lead_time > timedelta(hours=NEWS_PRE_WINDOW_HOURS):
        #     continue

        if lead_time <= timedelta(hours=NEWS_PRE_WINDOW_HOURS):
            update_news_event_status(
                event,
                'analyzing',
                f'Less than {NEWS_PRE_WINDOW_HOURS}h to release; analyzing because the signal window is still actionable.'
            )

            if best is None or et < best['event_time_utc']:
                best = event

            continue

        if best is None or et < best['event_time_utc']:
            best = event

    return best

def format_lead_time(delta):
    total = int(delta.total_seconds() // 60)
    return f"{total // 60}h {total % 60}m"

# FIX: News telegram now shows direction-only (no Entry/SL/TP) + historical pattern
def build_news_event_telegram(event, results, now):
    et = event.get('event_time_utc')
    lead = format_lead_time(et - now) if et else 'N/A'
    event_time_str = event.get('time', 'N/A')
    lines = [
        "📰 <b>DER-AI NEWS IMPACT SIGNAL</b>",
        f"📌 Event: {event.get('event')}",
        f"🕒 Event time: {event_time_str}",
        f"⏳ Sent {lead} before the release (once per event)",
        "",
    ]
    for symbol, a in results.items():
        sig = a.get('signal', 'SKIPPED')
        if sig in ('BUY', 'SELL'):
            sig_emoji = "🟢" if sig == 'BUY' else "🔴"
            lines.append(f"{sig_emoji} <b>{symbol}</b>: {sig}")
            # Show historical pattern if available
            hist_pattern = a.get('historical_pattern', '')
            if hist_pattern:
                lines.append(f"📚 Historical: {hist_pattern[:200]}")
            reason = (a.get('reasoning') or '').strip()
            if reason:
                lines.append(f"🧠 {reason[:400]}")
            lines.append("")
        elif sig == 'WAIT':
            lines.append(f"⚪ <b>{symbol}</b>: WAIT")
            why = (a.get('rejection_reason') or a.get('reasoning') or 'No actionable edge.').strip()
            if why:
                lines.append(f"🧠 {why[:300]}")
            lines.append("")
        else:
            lines.append(f"⚪ <b>{symbol}</b>: skipped ({a.get('reason', 'rate limit')})")
            lines.append("")
    return "\n".join(lines)

def run_news_analysis_cycle(event, all_data, symbols):
    eid = event.get('event_id') or f"{event.get('event')}|{event.get('currency')}|{event.get('time')}"

    # HARD TOKEN GUARD:
    # If this event already has results or was already marked sent, do not run AI again.
    if st.session_state.news_event_results.get(eid) or st.session_state.news_signal_sent.get(eid):
        update_news_event_status(
            event,
            'sent',
            'Already analyzed; skipping repeat AI run to save tokens.'
        )
        return True

    update_news_event_status(event, 'analyzing', 'AI pre-news impact analysis in progress...')

    results = {}

    for symbol in symbols:
        wait_for_groq_spacing()

        analysis = analyze_symbol_premium(symbol, all_data, news_override=[event])

        if not isinstance(analysis, dict):
            results[symbol] = {'signal': 'SKIPPED', 'reason': 'invalid result'}
            continue

        if 'error' in analysis:
            results[symbol] = {'signal': 'SKIPPED', 'reason': analysis.get('error')}
            continue

        if analysis.get('rejection_reason') == 'RATE_LIMIT':
            results[symbol] = {'signal': 'SKIPPED', 'reason': 'Groq rate limit'}
            continue

        results[symbol] = analysis

    st.session_state.news_event_results[eid] = {
        'event': event,
        'results': results,
        'analyzed_at': datetime.now()
    }

    valid = {
        s: a
        for s, a in results.items()
        if a.get('signal') in ('BUY', 'SELL', 'WAIT')
    }

    if not valid:
        update_news_event_status(
            event,
            'waiting',
            'Pre-news analysis failed (rate limit); retrying next cycle.'
        )
        return False

    # IMPORTANT:
    # Once valid AI results exist, mark the event as completed immediately.
    # Telegram delivery can fail, but we should NOT re-run the AI analysis.
    st.session_state.news_signal_sent[eid] = {
        'summary_sent': False,
        'symbols': list(valid),
        'analysis_completed': True,
        'completed_at': datetime.now()
    }

    update_news_event_status(
        event,
        'sent',
        'Pre-news analysis completed; will not re-analyze.'
    )

    message = build_news_event_telegram(event, results, datetime.now(timezone.utc))
    telegram_ok = send_telegram_message(message)

    if telegram_ok:
        st.session_state.news_signal_sent[eid]['summary_sent'] = True
        update_news_event_status(
            event,
            'sent',
            'Pre-news impact signal sent to Telegram (once).'
        )
    else:
        update_news_event_status(
            event,
            'sent',
            'Pre-news analysis completed; Telegram delivery failed/not configured, but AI will not repeat.'
        )

    return True
# ── Groq / Rate Limits ────────────────────────────────────────────────────────
GROQ_MIN_REQUEST_INTERVAL = 60
GROQ_TOKEN_LIMIT_PER_MINUTE = 12000
GROQ_ESTIMATED_RESPONSE_TOKENS = 600
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

def estimate_tokens_for_text(text):
    return max(1, int(len(text) / 4))

def estimate_analysis_tokens(system_prompt, user_content):
    prompt_text = system_prompt + ' ' + ' '.join([item.get('text', '') for item in user_content])
    return estimate_tokens_for_text(prompt_text) + GROQ_ESTIMATED_RESPONSE_TOKENS

def format_duration(seconds):
    return f"{int(seconds) // 60}m {int(seconds) % 60}s"

def reserve_gpt_tokens(estimated_tokens):
    now = datetime.now()
    window_start = st.session_state.gpt_token_window_start
    if (now - window_start).total_seconds() >= 60:
        st.session_state.gpt_token_window_start = now
        st.session_state.gpt_tokens_used = 0
    if estimated_tokens is None:
        estimated_tokens = 0
    if st.session_state.gpt_tokens_used + estimated_tokens > GROQ_TOKEN_LIMIT_PER_MINUTE:
        next_reset = st.session_state.gpt_token_window_start + timedelta(minutes=1)
        st.session_state.gpt_rate_limit_until = next_reset
        st.session_state.gpt_rate_limit_reason = f"Token budget exceeded: {st.session_state.gpt_tokens_used}/{GROQ_TOKEN_LIMIT_PER_MINUTE} used. Needs {estimated_tokens} more tokens and resets at {next_reset.strftime('%H:%M:%S')}."
        return False
    st.session_state.gpt_rate_limit_reason = ''
    st.session_state.gpt_tokens_used += estimated_tokens
    return True

def update_analysis_status(symbol=None, message=None, estimated_tokens=None, estimated_duration=None, next_pair=None):
    st.session_state.analysis_status = {'current_pair': symbol, 'message': message, 'estimated_tokens': estimated_tokens, 'estimated_duration': estimated_duration, 'next_pair': next_pair}

def render_analysis_status(container):
    status = st.session_state.get('analysis_status', {})
    if not status or not status.get('current_pair'):
        container.markdown("**Analysis status:** idle. Waiting for the next scheduled run or manual start.")
        return
    lines = [f"**Current Pair:** {status.get('current_pair')}", f"**Status:** {status.get('message', 'Running')}"]
    if status.get('estimated_tokens') is not None:
        lines.append(f"**Estimated Tokens:** {status.get('estimated_tokens')}")
    if status.get('estimated_duration') is not None:
        lines.append(f"**Estimated Duration:** {format_duration(status.get('estimated_duration'))}")
    if status.get('next_pair'):
        lines.append(f"**Next Pair:** {status.get('next_pair')}")
    lines.append(f"**Tokens used this window:** {st.session_state.gpt_tokens_used}/{GROQ_TOKEN_LIMIT_PER_MINUTE}")
    if st.session_state.gpt_rate_limit_until:
        lines.append(f"**Groq cooldown until:** {st.session_state.gpt_rate_limit_until.strftime('%H:%M:%S')}")
    container.markdown("\n".join(lines))

def is_scheduled_run_due():
    return (st.session_state.bot_running and st.session_state.next_check_time is not None and datetime.now() >= st.session_state.next_check_time and not st.session_state.analysis_in_progress)

def get_market_data(force_refresh=False):
    if not force_refresh and is_market_data_cached():
        return st.session_state.get('cached_market_data', fetch_all_data())
    data = fetch_all_data()
    if data and any(not df['M10'].empty or not df['H1'].empty or not df['H4'].empty for df in data.values()):
        st.session_state.cached_market_data = data
        st.session_state.last_market_fetch_time = datetime.now()
        return data
    if st.session_state.get('cached_market_data'):
        return st.session_state.cached_market_data
    st.session_state.cached_market_data = data
    st.session_state.last_market_fetch_time = datetime.now()
    return data

def is_market_data_cached(ttl_seconds=120):
    last_fetch = st.session_state.get('last_market_fetch_time')
    if not last_fetch:
        return False
    return (datetime.now() - last_fetch).total_seconds() < ttl_seconds

def is_gpt_rate_limited():
    retry_until = st.session_state.get('gpt_rate_limit_until')
    return retry_until is not None and datetime.now() < retry_until

def get_cached_analysis(symbol):
    cached = st.session_state.cached_analysis.get(symbol)
    if not cached:
        return None
    if cached.get('market_fetch_time') != st.session_state.get('last_market_fetch_time'):
        return None
    return cached.get('result')

def set_cached_analysis(symbol, analysis):
    st.session_state.cached_analysis[symbol] = {'market_fetch_time': st.session_state.get('last_market_fetch_time'), 'result': analysis}

def load_market_data(force_refresh=False):
    if not force_refresh and is_market_data_cached():
        st.info("💾 Using cached market data for instant refresh.")
        return get_market_data(force_refresh=False)
    with st.spinner("🔄 Loading market data... This should only take a few seconds."):
        return get_market_data(force_refresh=True)

MARKET_ANALYSIS_PROMPT = build_market_analysis_prompt()
NEWS_ANALYSIS_PROMPT = build_news_analysis_prompt()

def call_gpt(system_prompt, user_content, max_tokens=2000, retry_count=0, estimated_tokens=None):
    api_key = get_secret("GROQ_API_KEY", "")
    if not api_key or not api_key.startswith("gsk_"):
        return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": "Missing Groq API Key.", "estimated_tokens": estimated_tokens}
    if estimated_tokens is None:
        estimated_tokens = estimate_analysis_tokens(system_prompt, user_content)
    if not reserve_gpt_tokens(estimated_tokens):
        retry_time = st.session_state.gpt_rate_limit_until or (datetime.now() + timedelta(seconds=GROQ_MIN_REQUEST_INTERVAL))
        st.session_state.gpt_rate_limit_until = retry_time
        return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": "RATE_LIMIT", "rate_limit_reason": st.session_state.gpt_rate_limit_reason, "estimated_tokens": estimated_tokens}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for model in GROQ_MODELS:
        try:
            if is_gpt_rate_limited():
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": "RATE_LIMIT", "estimated_tokens": estimated_tokens}
            time_since_last = (datetime.now() - st.session_state.last_gpt_request_time).total_seconds() if st.session_state.last_gpt_request_time else None
            if time_since_last is not None and time_since_last < GROQ_MIN_REQUEST_INTERVAL:
                wait_time = int(GROQ_MIN_REQUEST_INTERVAL - time_since_last)
                st.session_state.gpt_rate_limit_until = datetime.now() + timedelta(seconds=wait_time)
                st.session_state.gpt_rate_limit_reason = f"Minimum request spacing not met. Wait {wait_time}s before the next Groq call."
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": "RATE_LIMIT", "rate_limit_reason": st.session_state.gpt_rate_limit_reason, "estimated_tokens": estimated_tokens}
            payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], "max_tokens": max_tokens, "temperature": 0.1}
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=90)
            st.session_state.last_gpt_request_time = datetime.now()
            if res.status_code == 429:
                retry_after = int(res.headers.get('Retry-After', '60')) if res.headers.get('Retry-After') else 60
                st.session_state.gpt_rate_limit_until = datetime.now() + timedelta(seconds=retry_after)
                st.session_state.gpt_rate_limit_reason = f"Groq 429 rate limit from API response. Retry after {retry_after}s."
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": "RATE_LIMIT", "rate_limit_reason": st.session_state.gpt_rate_limit_reason, "estimated_tokens": estimated_tokens}
            if res.status_code in (400, 404) and 'model' in res.text.lower():
                continue
            res_data = res.json()
            content = res_data['choices'][0]['message'].get('content', '')
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content).strip()
            content = re.sub(r',\s*([}\]])', r'\1', content)
            result = json.loads(content)
            result['model_used'] = model
            result['estimated_tokens'] = estimated_tokens
            return result
        except json.JSONDecodeError:
            continue
        except Exception as e:
            if model == GROQ_MODELS[-1]:
                return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": f"Error: {str(e)}", "estimated_tokens": estimated_tokens}
            continue
    return {"signal": "WAIT", "confluence_score": 0, "confidence": "LOW", "rejection_reason": "Error: all Groq models failed.", "estimated_tokens": estimated_tokens}

# ── Level Enforcement ─────────────────────────────────────────────────────────
def enforce_structural_math(analysis, swings, current_price, symbol, pair_config=None, atr=None):
    signal = analysis.get('signal')
    if signal not in ['BUY', 'SELL']:
        return analysis
    ai_sl = analysis.get('stop_loss')
    ai_tp_list = analysis.get('take_profit', [])
    ai_tp = ai_tp_list[0] if ai_tp_list else None
    pair_config = pair_config or get_pair_config(symbol)
    hard_min = current_price * pair_config.get('min_dist_pct', 0.001)
    floor = min(hard_min, 0.5 * atr) if atr else hard_min
    max_risk = current_price * pair_config.get('max_risk_pct', 0.01)
    target_rr = pair_config.get('target_rr', 1.2)
    ai_levels_valid = False
    if ai_sl and ai_tp:
        if signal == 'BUY':
            risk = current_price - ai_sl
            reward = ai_tp - current_price
            if risk + 1e-6 >= floor and risk + 1e-6 <= max_risk and risk > 0 and (reward / risk) + 1e-6 >= target_rr and ai_sl < current_price and ai_tp > current_price:
                ai_levels_valid = True
        elif signal == 'SELL':
            risk = ai_sl - current_price
            reward = current_price - ai_tp
            if risk + 1e-6 >= floor and risk + 1e-6 <= max_risk and risk > 0 and (reward / risk) + 1e-6 >= target_rr and ai_sl > current_price and ai_tp < current_price:
                ai_levels_valid = True
    if ai_levels_valid:
        analysis['stop_loss'] = round(ai_sl, 2)
        analysis['take_profit'] = [round(ai_tp, 2)]
        analysis['levels_source'] = 'AI'
    else:
        if signal == 'SELL':
            valid_sl = [h for h in swings.get('recent_swing_highs', []) if h > current_price]
            swing_dist = (valid_sl[0] - current_price) if valid_sl else floor
            sl_dist = min(max(swing_dist, floor), max_risk)
            analysis['stop_loss'] = round(current_price + sl_dist, 2)
            tp_dist = sl_dist * target_rr
            valid_tp = [l for l in swings.get('recent_swing_lows', []) if l < current_price]
            target_tp = current_price - tp_dist
            analysis['take_profit'] = [round(valid_tp[0], 2)] if valid_tp and target_tp < valid_tp[0] else [round(target_tp, 2)]
        elif signal == 'BUY':
            valid_sl = [l for l in swings.get('recent_swing_lows', []) if l < current_price]
            swing_dist = (current_price - valid_sl[0]) if valid_sl else floor
            sl_dist = min(max(swing_dist, floor), max_risk)
            analysis['stop_loss'] = round(current_price - sl_dist, 2)
            tp_dist = sl_dist * target_rr
            valid_tp = [h for h in swings.get('recent_swing_highs', []) if h > current_price]
            target_tp = current_price + tp_dist
            analysis['take_profit'] = [round(valid_tp[0], 2)] if valid_tp and target_tp > valid_tp[0] else [round(target_tp, 2)]
        analysis['levels_source'] = 'PYTHON'
    analysis['entry'] = round(current_price, 2)
    risk = abs(analysis['entry'] - analysis['stop_loss'])
    reward = abs(analysis['entry'] - analysis['take_profit'][0])
    analysis['rr_ratio'] = round(reward / risk, 2) if risk > 0 else 0
    return analysis

def normalize_market_levels(analysis, current_price, symbol, pair_config=None):
    if analysis.get('signal') not in ['BUY', 'SELL']:
        return analysis
    pair_config = pair_config or get_pair_config(symbol)
    max_entry_gap = current_price * pair_config.get('max_entry_gap_pct', 0.003)
    max_risk = current_price * pair_config.get('max_risk_pct', 0.01)
    entry = analysis.get('entry', current_price)
    if abs(entry - current_price) > max_entry_gap:
        analysis['entry'] = round(current_price, 2)
    sl = analysis.get('stop_loss')
    tp_list = analysis.get('take_profit', [])
    if not sl or not tp_list:
        return analysis
    if analysis['signal'] == 'BUY':
        if sl >= current_price:
            analysis['stop_loss'] = round(current_price - max_risk, 2)
        if analysis['take_profit'][0] - current_price > max_risk * 2.5:
            analysis['take_profit'] = [round(current_price + max_risk * 2.5, 2)]
    else:
        if sl <= current_price:
            analysis['stop_loss'] = round(current_price + max_risk, 2)
        if current_price - analysis['take_profit'][0] > max_risk * 2.5:
            analysis['take_profit'] = [round(current_price - max_risk * 2.5, 2)]
    risk = abs(analysis['entry'] - analysis['stop_loss'])
    reward = abs(analysis['entry'] - analysis['take_profit'][0])
    analysis['rr_ratio'] = round(reward / risk, 2) if risk > 0 else 0
    return analysis

def enforce_live_entry_proximity(analysis, current_price, symbol, pair_config=None, swings=None):
    if analysis.get('signal') not in ['BUY', 'SELL'] or current_price is None:
        return analysis
    pair_config = pair_config or get_pair_config(symbol)
    max_pts = pair_config.get('max_entry_points')
    max_gap_pct = pair_config.get('max_entry_gap_pct', 0.003)
    allowed = min(max_pts, current_price * max_gap_pct) if max_pts else current_price * max_gap_pct
    entry = analysis.get('entry', current_price)
    gap = abs(entry - current_price)
    if gap <= allowed:
        return analysis
    delta = entry - current_price
    analysis['confidence'] = 'MEDIUM'
    analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 72)
    analysis['entry'] = round(current_price, 2)
    if analysis.get('stop_loss'):
        analysis['stop_loss'] = round(analysis['stop_loss'] - delta, 2)
    if analysis.get('take_profit'):
        analysis['take_profit'] = [round(t - delta, 2) for t in analysis['take_profit']]
    analysis['order_type'] = 'MARKET'
    analysis['rejection_reason'] = None
    analysis['reasoning'] = f"The original entry was {gap:.2f} points away from the live price, so the setup has been re-anchored to {current_price:.2f} (SL/TP shifted to preserve the intended risk/reward)."
    risk = abs(analysis['entry'] - analysis['stop_loss'])
    reward = abs(analysis['entry'] - analysis['take_profit'][0])
    analysis['rr_ratio'] = round(reward / risk, 2) if risk > 0 else 0.0
    return analysis

def enforce_max_entry_distance(analysis, current_price, symbol, pair_config=None):
    if analysis.get('signal') not in ('BUY', 'SELL') or current_price is None:
        return analysis
    pair_config = pair_config or get_pair_config(symbol)
    max_pts = pair_config.get('max_entry_points')
    if not max_pts:
        return analysis
    entry = analysis.get('entry', current_price)
    gap = abs(entry - current_price)
    if gap <= max_pts:
        return analysis
    delta = entry - current_price
    analysis['entry'] = round(current_price, 2)
    if analysis.get('stop_loss'):
        analysis['stop_loss'] = round(analysis['stop_loss'] - delta, 2)
    if analysis.get('take_profit'):
        analysis['take_profit'] = [round(t - delta, 2) for t in analysis['take_profit']]
    analysis['order_type'] = 'MARKET'
    analysis['reasoning'] = f"{analysis.get('reasoning', '')} Entry was re-anchored to the live price ({current_price:.2f}) to respect the {max_pts}-point maximum entry distance for {symbol}; SL/TP were shifted to preserve the intended risk/reward."
    risk = abs(analysis['entry'] - analysis['stop_loss'])
    reward = abs(analysis['entry'] - analysis['take_profit'][0])
    analysis['rr_ratio'] = round(reward / risk, 2) if risk > 0 else 0
    return analysis

def apply_dxy_guardrails(analysis, symbol, dxy_context):
    if symbol not in ['XAUUSD', 'EURUSD', 'BTCUSD'] or not dxy_context:
        return analysis
    trend = dxy_context.get('trend')
    price_vs_vwap = dxy_context.get('price_vs_vwap')
    reasoning = (analysis.get('reasoning') or '').lower()
    signal = analysis.get('signal')
    if trend == 'BULLISH' and price_vs_vwap == 'ABOVE':
        expected_bias = 'SELL'
    elif trend == 'BEARISH' and price_vs_vwap == 'BELOW':
        expected_bias = 'BUY'
    else:
        expected_bias = None
    if expected_bias is None:
        analysis['dxy_correlation'] = 'NEUTRAL'
        return analysis
    if signal == expected_bias:
        analysis['dxy_correlation'] = 'CONFIRMING'
        if 'dxy' not in reasoning and 'dollar' not in reasoning:
            analysis['reasoning'] = f"{analysis.get('reasoning', '')} DXY is confirming the directional bias because the dollar index is {trend.lower()} and price is {price_vs_vwap.lower()} VWAP."
        return analysis
    analysis['dxy_correlation'] = 'CONTRADICTING'
    if analysis.get('confidence') == 'HIGH':
        analysis['confidence'] = 'MEDIUM'
    analysis['confluence_score'] = max(0, analysis.get('confluence_score', 0) - 4)
    if 'dxy' not in reasoning and 'dollar' not in reasoning:
        analysis['reasoning'] = f"{analysis.get('reasoning', '')} The setup is contrarian versus the DXY bias, so it needs an explicit macro explanation to justify the trade."
    return analysis

def apply_htf_trend_guard(analysis, symbol, htf_context):
    if analysis.get('signal') not in ['BUY', 'SELL'] or not isinstance(htf_context, dict):
        return analysis
    trend = str(htf_context.get('trend') or '').upper()
    bias = str(htf_context.get('bias') or '').upper()
    if trend not in {'BULLISH', 'BEARISH'} and bias not in {'BULLISH', 'BEARISH'}:
        return analysis
    expected_bias = trend or bias
    signal = analysis.get('signal')
    if (signal == 'BUY' and expected_bias == 'BEARISH') or (signal == 'SELL' and expected_bias == 'BULLISH'):
        analysis['confidence'] = 'MEDIUM' if analysis.get('confidence') == 'HIGH' else analysis.get('confidence')
        analysis['confluence_score'] = min(analysis.get('confluence_score', 0), 78)
        analysis['reasoning'] = f"{analysis.get('reasoning', '')} Note: the higher-timeframe trend is {expected_bias.lower()}, so this countertrend idea carries reduced conviction and needs strong structural confirmation."
    return analysis

# FIX: For news signals, direction-only (no Entry/SL/TP). Direction comes from firm desk bias.
def build_pre_news_fallback_analysis(symbol, m10, swings, pair_config, dxy_context, news_context, candles=None, phase_context=None, live_price=None, htf_context=None, picture=None, firm=None, firm_notes=None, learning=None):
    """Python-only fallback for news signals. Direction-only (no Entry/SL/TP)."""
    current_price = live_price if live_price is not None else m10['Close'].iloc[-1]
    micro = calculate_microstructure(m10)
    phase_context = phase_context or detect_market_phase(m10, swings=swings)
    if picture is None:
        picture = build_mtf_picture({'M10': m10}, symbol)
    if firm is None:
        firm, firm_notes = resolve_firm_direction(symbol, picture)
    confluence = detect_directional_confluence(m10, swings=swings, htf_context=htf_context, dxy_context=dxy_context, symbol=symbol)
    # For news: firm desk bias is the only direction authority
    direction = firm
    learn_note, learn_adj = learning if learning else (None, 0)
    if phase_context.get('phase') == 'exhaustion':
        direction = None
    
    if direction is None:
        direction = confluence.get('direction')
    if direction is None:
        direction = 'BUY' if micro.get('momentum') == 'BULLISH' else 'SELL'
    expected_signal = direction
    bias = 'BULLISH' if direction == 'BUY' else 'BEARISH'
    supporting = confluence.get('bullish_evidence') if direction == 'BUY' else confluence.get('bearish_evidence')
    reasoning_parts = []
    reasoning_parts.append(f"Directional evidence audit supports {direction}: " + '; '.join(supporting[:4]) + '.')
    if learn_note:
        reasoning_parts.append(learn_note)
    if firm_notes:
        reasoning_parts.extend(firm_notes)
    reasoning = ' '.join(reasoning_parts)
    score = max(MINIMUM_CONFLUENCE_SCORE, min(82, 74 + (2 if abs(picture.get('score', 0)) >= 3 else 0) + learn_adj))
    # News signals: direction-only, no Entry/SL/TP
    return {'bias': bias, 'signal': expected_signal, 'confluence_score': score, 'confidence': 'MEDIUM', 'dxy_correlation': 'CONFIRMING' if dxy_context else 'NEUTRAL', 'microstructure_read': f"VWAP {micro.get('price_vs_vwap', 'NEUTRAL')} | RVOL {micro.get('rvol', 0):.2f} | Momentum {micro.get('momentum', 'NEUTRAL')}", 'pre_news_bias': news_context.get('pre_news_bias', 'News-driven reaction expected'), 'reasoning': reasoning, 'rejection_reason': None, 'structural_score': 72, 'score_reason': 'Fallback desk model using the MTF directional picture.', 'candidate_direction': expected_signal, 'levels_source': 'PYTHON'}

# ── Main Analysis Engine ──────────────────────────────────────────────────────
def analyze_symbol_premium(symbol, all_data, news_override=None):
    try:
        data = all_data.get(symbol, {})
        m10 = data.get('M10', pd.DataFrame())
        h1 = data.get('H1', pd.DataFrame())
        h4 = data.get('H4', pd.DataFrame())
        live_snapshot = get_live_market_snapshot(symbol, YFINANCE_MAP.get(symbol, symbol), fallback_df=m10)
        if m10.empty:
            return {"error": f"Failed to fetch market data for {symbol}. Yahoo Finance may be temporarily rate-limiting your IP. Please wait a few minutes and try again."}
        micro = calculate_microstructure(m10)
        current_price = live_snapshot.get('price') or float(m10['Close'].iloc[-1])
        resolve_pending_outcomes({symbol: current_price})
        swings = find_swings(m10)
        pair_config = get_pair_config(symbol)
        max_entry_distance = f"{pair_config.get('max_entry_points', 10)} points (HARD LIMIT for {symbol})"
        dxy_data = all_data.get('DXY', {}).get('H1', pd.DataFrame())
        dxy_summary = "DXY Data Unavailable"
        dxy_context = None
        if not dxy_data.empty:
            dxy_price = dxy_data['Close'].iloc[-1]
            dxy_micro = calculate_microstructure(dxy_data)
            dxy_context = {'trend': dxy_micro['momentum'], 'price_vs_vwap': dxy_micro['price_vs_vwap']}
            dxy_summary = f"Current: {dxy_price} | VWAP Position: {dxy_micro['price_vs_vwap']} | Momentum: {dxy_micro['momentum']} | RVOL: {dxy_micro['rvol']}"
        htf_context = None
        if not h4.empty:
            h4_micro = calculate_microstructure(h4)
            htf_context = {'trend': h4_micro['momentum'], 'bias': h4_micro['momentum'], 'price_vs_vwap': h4_micro['price_vs_vwap']}
        picture = build_mtf_picture(all_data, symbol)
        firm = None
        firm_notes = []
        if picture:
            firm, firm_notes = resolve_firm_direction(symbol, picture)
        m15_data = data.get('M15', pd.DataFrame())
        m30_data = data.get('M30', pd.DataFrame())
        if news_override:
            news_payload = [{'event_time_utc': n.get('event_time_utc', datetime.now(timezone.utc) + timedelta(minutes=max(0, n.get('minutes_until', 0)))), 'within_2h': n.get('within_2h', False), 'time': n.get('time', ''), 'currency': n.get('currency', ''), 'event': n.get('event', '')} for n in news_override]
            nc = build_news_context(news_payload, reference_dt=datetime.now(timezone.utc))
            news_text = format_news_summary(news_payload, limit=len(news_payload))
            news_context = {'within_2h': nc.get('within_2h', False), 'bias': nc.get('bias', 'neutral'), 'upcoming_count': nc.get('upcoming_count', 0), 'next_event': nc.get('next_event'), 'pre_news_bias': nc.get('pre_news_bias', 'No imminent high-impact event.')}
            prompt_template = NEWS_ANALYSIS_PROMPT
        else:
            news_context = {'within_2h': False, 'bias': 'neutral', 'upcoming_count': 0, 'next_event': None, 'pre_news_bias': 'No imminent high-impact event in the next 2 hours.'}
            news_text = 'No event-driven news context is considered for this live market analysis.'
            prompt_template = MARKET_ANALYSIS_PROMPT
        phase_context = detect_market_phase(m10, swings=swings)
        setup_context = build_setup_context(m10, swings, current_price, symbol, dxy_context=dxy_context, news_context=news_context)
        structural_context = calculate_structural_score(m10, symbol, dxy_context=dxy_context, news_context=news_context, phase_context=phase_context)
        candles = analyze_candle_structure(m10)
        bos, choch = detect_bos_choch(m10)
        order_blocks = detect_order_blocks(m10)
        fvgs = detect_fvg(m10)
        sweeps = detect_liquidity_sweeps(m10)
        structure_parts = []
        if bos or choch:
            structure_parts.append(f"BOS/CHOCH: {bos or choch}")
        if order_blocks:
            structure_parts.append("Order blocks: " + ", ".join([f"{ob['type']}@{ob['price']:.2f}" for ob in order_blocks]))
        if fvgs:
            structure_parts.append("FVGs: " + ", ".join([f"{fvg['type']}({fvg['top']:.2f}->{fvg['bottom']:.2f})" for fvg in fvgs]))
        if sweeps:
            structure_parts.append("Sweeps: " + ", ".join([f"{s['type']}@{s['price']:.2f}" for s in sweeps]))
        if candles:
            structure_parts.append("Recent candles: " + "; ".join([f"{c['time'].strftime('%H:%M')} {c['pattern']} ({c['candle_type']})" for c in candles]))
        structure_context = " | ".join(structure_parts) if structure_parts else "No strong structural clues detected in the provided intrabar data."
        market_structure_summary = build_market_structure_summary(m10, current_price=current_price, swings=swings, order_blocks=order_blocks, fvgs=fvgs, sweeps=sweeps, bos=bos, choch=choch, symbol=symbol)
        multitimeframe_context = build_multitimeframe_context(all_data, symbol)
        rsi_values = build_rsi_values_context(all_data, symbol)
        premium_discount = build_premium_discount_context(m10, current_price)
        volatility_context = build_volatility_context(m10)
        ledger = detect_directional_confluence(m10, swings=swings, htf_context=htf_context, dxy_context=dxy_context, symbol=symbol)
        directional_ledger = f"Bullish ({ledger['bull_count']}): {'; '.join(ledger['bullish_evidence']) or 'none'} | Bearish ({ledger['bear_count']}): {'; '.join(ledger['bearish_evidence']) or 'none'} | Ledger direction: {ledger['direction'] or 'none'}"
        rsi_context = ''
        if not m10.empty:
            divergence = detect_rsi_divergence(m10)
            rsi_context = f"M10 RSI context: {divergence['type']} - {divergence['reason']}" if divergence else 'M10 RSI context: no clear divergence detected.'
            for label, frame in [('M15', m15_data), ('M30', m30_data), ('H1', h1)]:
                if frame is not None and not getattr(frame, 'empty', True):
                    d = detect_rsi_divergence(frame)
                    if d:
                        rsi_context += f" | {label} RSI context: {d['type']} - {d['reason']}"
        if not rsi_context:
            rsi_context = 'RSI context unavailable.'
        h1_summary = f"Latest H1 close: {h1['Close'].iloc[-1]:.2f}" if not h1.empty else "H1 data unavailable"
        h4_summary = f"Latest H4 close: {h4['Close'].iloc[-1]:.2f}" if not h4.empty else "H4 data unavailable"
        htf_summary = f"H1: {h1_summary} | H4: {h4_summary}"
        prompt_data = f"Symbol: {symbol} | Live Price: {current_price} | Swing Highs: {swings['recent_swing_highs']} | Swing Lows: {swings['recent_swing_lows']} | Market Phase: {phase_context['phase']} | Phase Reason: {phase_context['reason']} | Setup Type: {setup_context['setup_type']} | Entry Timing: {setup_context['entry_timing']} | Entry Quality: {phase_context['entry_quality']} | Entry Rule: use a price-near entry and do not chase a distant level."
        prompt_micro = f"VWAP: {micro.get('vwap', 'N/A')} | Price vs VWAP: {micro.get('price_vs_vwap', 'N/A')} | RVOL: {micro.get('rvol', 'N/A')} ({micro.get('volume_anomaly', 'N/A')})"
        structural_score_context = f"Python structural score: {structural_context['structural_score']}/100 | Basis: {structural_context['score_reason']}"
        historical_context = build_historical_context(m10)
        firm_bias_text = f"{firm} (standing desk bias; weighted MTF evidence {picture.get('score', 0):+.1f})" if firm else "NONE - evidence tied; stand aside unless a clear edge emerges."
        all_format_kwargs = {
            'data_summary': prompt_data, 'microstructure_data': prompt_micro,
            'structure_context': structure_context, 'market_structure_summary': market_structure_summary,
            'multitimeframe_context': multitimeframe_context, 'rsi_values': rsi_values,
            'rsi_context': rsi_context, 'premium_discount': premium_discount,
            'volatility_context': volatility_context, 'htf_context': htf_summary,
            'dxy_data': dxy_summary, 'historical_context': historical_context,
            'news_summary': news_text, 'structural_score_context': structural_score_context,
            'directional_ledger': directional_ledger, 'firm_bias': firm_bias_text,
            'max_entry_distance': max_entry_distance,
        }

        try:
            prompt_text = prompt_template.format(**all_format_kwargs)
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            prompt_text = prompt_template.format(**{**all_format_kwargs, missing_key: f"[missing:{missing_key}]"})
        user_content = [{"type": "text", "text": prompt_text}]
        estimated_tokens = estimate_analysis_tokens("You are an Elite Macro Analyst. Output ONLY valid JSON.", user_content)
        analysis = call_gpt("You are an Elite Macro Analyst. Output ONLY valid JSON.", user_content, max_tokens=2000, estimated_tokens=estimated_tokens)
        analysis.setdefault('microstructure_read', prompt_micro)
        analysis.setdefault('rsi_context', rsi_context)
        analysis.setdefault('dxy_summary', dxy_summary)
        analysis.setdefault('live_price', current_price)
        analysis['setup_context'] = setup_context
        analysis['market_state'] = analysis.get('market_state') or setup_context['setup_type']
        update_market_state(analysis.get('market_state') or setup_context['setup_type'])
        if analysis.get('signal') == 'WAIT' and (analysis.get('rejection_reason') in {'Missing Groq API Key.', 'RATE_LIMIT'} or not analysis.get('rejection_reason')):
            analysis = build_pre_news_fallback_analysis(symbol, m10, swings, pair_config, dxy_context, news_context, candles=candles, phase_context=phase_context, live_price=current_price, htf_context=htf_context)
        analysis['estimated_tokens'] = analysis.get('estimated_tokens', estimated_tokens)
        # ── NEWS SIGNAL PATH: direction-only, skip entry/SL/TP enforcement ──
        if news_override:
            analysis['structural_score'] = structural_context['structural_score']
            analysis['atr'] = setup_context.get('atr')
            analysis['score_reason'] = structural_context['score_reason']
            analysis['candidate_direction'] = structural_context['candidate_direction']
            analysis['is_news_signal'] = True
            analysis = apply_dxy_guardrails(analysis, symbol, dxy_context)
            # Skip entry/SL/TP enforcement for news signals - direction only
            analysis['validation_detail'] = build_validation_detail(analysis, swings, current_price, symbol, pair_config=pair_config, structural_context=structural_context)
            analysis['display_reasoning'] = build_display_reason(analysis, symbol, current_price=current_price, phase_context=phase_context, structural_context=structural_context, dxy_context=dxy_context)
            analysis['symbol'] = symbol
            analysis['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            analysis['news_event'] = news_override[0].get('event', '')
            analysis['news_time'] = news_override[0].get('time', '')
            analysis['news_event_id'] = news_override[0].get('event_id')
            return analysis
        # ── REGULAR SIGNAL PATH: full entry/SL/TP enforcement ──
        if setup_context['setup_type'] == 'exhaustion':
            analysis['signal'] = 'WAIT'
            analysis['confidence'] = 'LOW'
            analysis['confluence_score'] = min(analysis.get('confluence_score', 0), MINIMUM_CONFLUENCE_SCORE)
            analysis['rejection_reason'] = 'Exhaustion is already visible, so the market is too extended to justify forcing a fresh trade into the move.'
        if setup_context['entry_timing'] == 'late':
            analysis['signal'] = 'WAIT'
            analysis['confidence'] = 'LOW'
            analysis['confluence_score'] = min(analysis.get('confluence_score', 0), MINIMUM_CONFLUENCE_SCORE)
            analysis['rejection_reason'] = 'The entry is already late, the move is in progress, and the structure is no longer offering a clean early re-entry opportunity.'
        analysis = apply_htf_trend_guard(analysis, symbol, htf_context)
        analysis = enforce_live_entry_proximity(analysis, current_price, symbol, pair_config=pair_config, swings=swings)
        ai_score = int(round(analysis.get('confluence_score', 0)))
        analysis['confluence_score'] = min(100, max(0, ai_score))
        if analysis.get('confidence') == 'LOW' and analysis['confluence_score'] >= 75:
            analysis['confidence'] = 'MEDIUM'
        elif analysis.get('confidence') == 'MEDIUM' and analysis['confluence_score'] >= 85:
            analysis['confidence'] = 'HIGH'
        analysis['structural_score'] = structural_context['structural_score']
        analysis['atr'] = setup_context.get('atr')
        analysis['score_reason'] = structural_context['score_reason']
        analysis['candidate_direction'] = structural_context['candidate_direction']
        analysis['is_news_signal'] = bool(news_override)
        analysis = apply_dxy_guardrails(analysis, symbol, dxy_context)
        analysis = cross_check_ai_evidence(analysis)
        # M10 ledger correction runs FIRST...
        analysis = apply_direction_correction_guard(analysis, ledger, symbol)
        analysis = cross_check_ai_evidence(analysis)
        # ...then the standing desk bias has the FINAL word.
        if firm and analysis.get('signal') in ('BUY', 'SELL') and analysis['signal'] != firm:
            ev = analysis.get('directional_evidence') or {}
            counter = len(ev.get('bullish', [])) if firm == 'BUY' else len(ev.get('bearish', []))
            if counter >= 3:
                analysis['reasoning'] = (analysis.get('reasoning') or '') + f" (AI overrode the standing {firm} desk bias with {counter} counter-evidences.)"
            else:
                analysis['signal'] = firm
                analysis['reasoning'] = (analysis.get('reasoning') or '') + f" The standing {firm} desk bias is maintained; the AI view was aligned to the desk bias."
        analysis = apply_conservative_signal_filter(analysis, structural_context, candles, dxy_context, news_context, current_price, swings, symbol, pair_config=pair_config)
        analysis = enforce_structural_math(analysis, swings, current_price, symbol, pair_config=pair_config, atr=setup_context.get('atr'))
        if analysis.get('signal') in ('BUY', 'SELL') and (not analysis.get('stop_loss') or not analysis.get('take_profit')):
            execution_plan = build_execution_plan(m10, current_price, analysis.get('signal'), swings, pair_config=pair_config, atr=setup_context.get('atr'))
            analysis['entry'] = execution_plan['entry']
            analysis['stop_loss'] = execution_plan['stop_loss']
            analysis['take_profit'] = execution_plan['take_profit']
            analysis['risk_band'] = execution_plan['risk_band']
            analysis['levels_source'] = 'PYTHON'
        else:
            analysis['risk_band'] = round(abs(analysis.get('entry', 0) - analysis.get('stop_loss', 0)), 2)
        if not news_override:
            analysis = normalize_market_levels(analysis, current_price, symbol, pair_config=pair_config)
        analysis = enforce_max_entry_distance(analysis, current_price, symbol, pair_config=pair_config)
        if analysis.get('signal') in ('BUY', 'SELL') and analysis.get('take_profit'):
            final_note = f"Final levels: Entry {analysis['entry']} | SL {analysis['stop_loss']} | TP {analysis['take_profit'][0]}."
            analysis['order_description'] = f"{final_note} {analysis.get('order_description') or ''}".strip()
        analysis['validation_detail'] = build_validation_detail(analysis, swings, current_price, symbol, pair_config=pair_config, structural_context=structural_context)
        analysis['display_reasoning'] = build_display_reason(analysis, symbol, current_price=current_price, phase_context=phase_context, structural_context=structural_context, dxy_context=dxy_context)
        analysis['symbol'] = symbol
        analysis['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if news_override:
            analysis['news_event'] = news_override[0].get('event', '')
            analysis['news_time'] = news_override[0].get('time', '')
            analysis['news_event_id'] = news_override[0].get('event_id')
        return analysis
    except Exception as e:
        debug_log(f"analyze_symbol_premium exception for {symbol}: {str(e)}", exc_info=True)
        return {"error": str(e)}

# ── Main App UI ───────────────────────────────────────────────────────────────
st.title("🌍 Der-AI | Quantitative Macro System")
st.markdown("**DXY Correlation | VWAP Microstructure | Pre-News Positioning | Telegram Bridge**")
st.sidebar.header("⚙️ System Configuration")
selected_symbols = st.sidebar.multiselect("Monitor Symbols", SYMBOLS, default=['XAUUSD', 'EURUSD', 'BTCUSD', 'US30'])
st.sidebar.info(f"Analysis interval is fixed at {ANALYSIS_INTERVAL_MINUTES} minutes.")
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Signal Sensitivity")
st.sidebar.caption(f"Signals are only accepted when the AI score reaches at least {MINIMUM_CONFLUENCE_SCORE}/100.")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("▶️ START", type="primary", use_container_width=True):
        st.session_state.bot_running = True
        st.session_state.next_check_time = datetime.now()
        st.session_state.rate_limit_hit = False
        add_notification('info', "✅ Bot started. Monitoring markets.")
        st.rerun()
with col2:
    if st.button("⏹️ STOP", type="secondary", use_container_width=True):
        st.session_state.bot_running = False
        add_notification('warning', "⏸️ Bot stopped by user.")
        st.rerun()
with col3:
    if st.button("🗑️ CLEAR", use_container_width=True):
        st.session_state.active_signals = {}
        st.session_state.signal_history = []
        clear_notifications()
        st.session_state.bot_running = False
        st.session_state.last_analysis_time = None
        st.session_state.next_check_time = None
        st.session_state.rate_limit_hit = False
        add_notification('info', "🗑️ System memory cleared.")
        st.rerun()
if st.session_state.bot_running:
    st.sidebar.success("✅ **BOT RUNNING**")
    if st.session_state.next_check_time:
        time_left = (st.session_state.next_check_time - datetime.now()).total_seconds()
        if time_left > 0:
            st.sidebar.markdown(f"⏱️ Next check in: <strong id='derai-sidebar-countdown'>{int(time_left // 60)}m {int(time_left % 60)}s</strong>", unsafe_allow_html=True)
        else:
            st.sidebar.info("⏱️ Checking now...")
else:
    st.sidebar.warning("⏸️ **BOT STOPPED**")
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Stats")
st.sidebar.metric("Premium Signals", len(st.session_state.signal_history))
st.sidebar.metric("Notifications", len(get_notifications()))
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔴 Live Monitoring", "📜 Signal History", "🔔 Notifications", "📰 News Calendar", "⚙️ Settings"])
with tab1:
    st.header("🔴 Live Multi-Timeframe Macro Analysis")
    if st.session_state.bot_running:
        if st.session_state.next_check_time:
            time_left = (st.session_state.next_check_time - datetime.now()).total_seconds()
            if time_left > 0:
                mins, secs = int(time_left // 60), int(time_left % 60)
                next_time_str = st.session_state.next_check_time.strftime('%H:%M:%S')
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h2 style="margin: 0; font-size: 1.2em; font-weight: normal;">⏱️ Next Live Analysis In</h2>
                    <h1 id="derai-countdown-text" style="margin: 10px 0; font-size: 2.5em; font-weight: bold; color: #bfdbfe;">{mins}m {secs}s</h1>
                    <p style="margin: 5px 0 0 0; font-size: 1em; opacity: 0.95;">Scheduled at <strong id="derai-scheduled-at">{next_time_str}</strong></p>
                    <p style="margin: 10px 0 0 0; font-size: 0.95em; opacity: 0.9;">✅ Continuously fetching real-time market data, DXY correlation, and microstructure...</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(90deg, #059669 0%, #10b981 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                    <h2 style="margin: 0; font-size: 1.5em; font-weight: bold;">🔄 Running Live Analysis Now...</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Fetching latest multi-timeframe data...</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='background-color: #dc3545; color: white; padding: 15px; border-radius: 8px; text-align: center;'><h3>⚪ SYSTEM INACTIVE - Click START to begin</h3></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: #dc3545; color: white; padding: 15px; border-radius: 8px; text-align: center;'><h3>⚪ SYSTEM INACTIVE - Click START to begin</h3></div>", unsafe_allow_html=True)
    if is_market_data_cached():
        age = int((datetime.now() - st.session_state['last_market_fetch_time']).total_seconds())
        st.info(f"💾 Cached market data available ({age}s old). Repeat refreshes are near-instant.")
    render_countdown_timer()
    def process_symbol_result(result, symbol, is_auto=False, all_data=None):
        if not isinstance(result, dict):
            msg = f"❌ **{symbol}**: Analysis returned invalid result type ({type(result).__name__})."
            if not is_auto:
                st.error(msg)
            add_notification('warning', msg, symbol=symbol)
            return
        result['symbol'] = result.get('symbol') or symbol
        summary_reason = (result.get('display_reasoning') or result.get('reasoning') or result.get('rejection_reason') or result.get('error') or 'No additional details provided.')
        add_notification('info', f"🧠 **{symbol}**: AI analysis complete. Signal: {result.get('signal', 'N/A')} | Confidence: {result.get('confidence', 'N/A')} | Score: {result.get('confluence_score', 'N/A')}/100. Reason: {summary_reason}", symbol=symbol, signal=result.get('signal'), score=result.get('confluence_score'))
        if result.get('rejection_reason') == 'RATE_LIMIT':
            st.session_state.rate_limit_hit = True
            next_retry = st.session_state.gpt_rate_limit_until or (datetime.now() + timedelta(seconds=GROQ_MIN_REQUEST_INTERVAL))
            st.session_state.next_check_time = next_retry
            reason = result.get('rate_limit_reason') or 'Groq is currently unavailable.'
            msg = f"⏳ **{symbol}**: Groq rate limit active. {reason} Pausing analysis until {next_retry.strftime('%H:%M:%S')} to protect API quota."
            if not is_auto:
                st.warning(msg)
            add_notification('warning', msg, symbol=symbol)
            return
        if 'error' in result:
            error_msg = result.get('error', 'Data fetch failed or no data available.')
            msg = f"❌ **{symbol}**: Error analyzing {symbol}: {error_msg}"
            if not is_auto:
                st.error(msg)
            add_notification('warning', msg, symbol=symbol)
            return
        is_valid_logic, logic_reason = validate_ai_logic(result)
        if not is_valid_logic:
            detail = result.get('validation_detail', '')
            msg = f"⚪ **{symbol}**: Signal Rejected. AI Logic Flaw: {logic_reason}"
            if detail:
                msg += f" | Structural Detail: {detail}"
            if not is_auto: st.info(msg)
            add_notification('warning', msg, symbol=symbol, signal=result.get('signal'))
            return
        if result.get('signal') == 'WAIT':
            ai_reason = result.get('display_reasoning') or result.get('reasoning') or result.get('rejection_reason', 'Market conditions do not meet high-confidence criteria.')
            detail = result.get('validation_detail', '')
            msg = f"⚪ **{symbol}**: No Trade (WAIT). AI Reason: {ai_reason}"
            if detail:
                msg += f" | Structural Detail: {detail}"
            if not is_auto:
                st.info(msg)
            add_notification('info', msg, symbol=symbol, signal='WAIT', score=result.get('confluence_score'))
            return
        pair_config = get_pair_config(symbol)
        is_valid_math, math_reason = validate_signal_math(result, pair_config=pair_config)
        if not is_valid_math:
            detail = result.get('validation_detail', '')
            msg = f"⚪ **{symbol}**: Signal Rejected. AI Reason: {math_reason}"
            if detail:
                msg += f" | Structural Detail: {detail}"
            if not is_auto: st.info(msg)
            add_notification('warning', msg, symbol=symbol, signal=result.get('signal'))
            return
        min_score = max(MINIMUM_CONFLUENCE_SCORE, pair_config.get('score_floor', MINIMUM_CONFLUENCE_SCORE))
        combined_score = result.get('confluence_score', 0)
        candidate_threshold = max(MINIMUM_CONFLUENCE_SCORE, pair_config.get('candidate_score', MINIMUM_CONFLUENCE_SCORE))
        if combined_score >= candidate_threshold and result.get('confidence') in ['HIGH', 'MEDIUM', 'LOW']:
            is_repeat = False
            current_time = datetime.now()
            if symbol in st.session_state.active_signals:
                last_sig = st.session_state.active_signals[symbol]
                time_diff_minutes = (current_time - last_sig['timestamp']).total_seconds() / 60
                entry_price = result.get('entry', 0)
                last_entry = last_sig['entry']
                previous_score = last_sig.get('score', 0)
                if (last_sig['direction'] == result.get('signal') and entry_price > 0 and last_entry > 0 and abs(last_entry - entry_price) / entry_price < 0.01 and time_diff_minutes < pair_config.get('cooldown_minutes', 30) and combined_score <= previous_score + 3):
                    is_repeat = True
            if is_repeat:
                last_time = st.session_state.active_signals[symbol]['timestamp'].strftime('%H:%M')
                msg = f"⏸️ **{symbol}**: Setup already active since {last_time}. Waiting for execution or structural invalidation. ({pair_config.get('cooldown_minutes', 30)}-min cooldown)"
                if not is_auto: st.info(msg)
                add_notification('info', msg, symbol=symbol, signal=result.get('signal'))
            elif combined_score >= min_score:
                sig_color = "🟢" if result.get('signal') == "BUY" else "🔴"
                if not is_auto: st.markdown(f"### {sig_color} **NEW SIGNAL:** {result.get('symbol', symbol)} - {result.get('signal')}")
                else: st.success(f"{sig_color} **NEW SIGNAL:** {result.get('symbol', symbol)} - {result.get('signal')} | Score: {result.get('confluence_score')}/100")
                if not is_auto:
                    st.write(f"**DXY Correlation:** {result.get('dxy_correlation', 'N/A')}")
                    st.write(f"**Microstructure:** {result.get('microstructure_read', 'N/A')}")
                    st.write(f"**Pre-News Bias:** {result.get('pre_news_bias', 'N/A')}")
                    st.write(f"**Levels source:** {result.get('levels_source', 'AI')}")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Bias", result.get('bias', 'N/A'))
                    col_b.metric("Confidence", result.get('confidence', 'N/A'))
                    col_c.metric("Score", f"{result.get('confluence_score', 0)}/100")
                    col_d.metric("R:R", f"1:{result.get('rr_ratio', 0)}")
                # For news signals: show direction-only (no Entry/SL/TP)
                if result.get('is_news_signal'):
                    st.info(f"**Direction:** {result.get('signal')}")
                    st.info(f"**Reasoning:** {result.get('reasoning')}")
                    if result.get('historical_pattern'):
                        st.info(f"**Historical Pattern:** {result.get('historical_pattern')}")
                else:
                    st.info(f"**Entry:** {result.get('entry')} | **SL:** {result.get('stop_loss')} | **TP:** {result.get('take_profit')}")
                if result.get('order_type'):
                    st.write(f"**Order Type:** {result.get('order_type')}")
                if result.get('order_description'):
                    st.write(f"**Execution Plan:** {result.get('order_description')}")
                if result.get('confluence_breakdown'):
                    st.write(f"**Confluence Breakdown:** {result.get('confluence_breakdown')}")
                st.write(f"**Reasoning:** {result.get('reasoning')}")
                st.session_state.active_signals[symbol] = {'direction': result.get('signal'), 'entry': result.get('entry', 0), 'timestamp': current_time, 'score': combined_score}
                result['analyzed_at'] = current_time
                st.session_state.signal_history.append(result)
                st.session_state.signal_ledger.append({'symbol': symbol, 'direction': result.get('signal'), 'entry': result.get('entry', 0), 'time': current_time, 'threshold': 0.5 * abs(result.get('entry', 0) - result.get('stop_loss', 0)) or 1.0})
                msg = build_telegram_signal_message(symbol, result)
                if send_telegram_message(msg) and not is_auto:
                    st.success("✅ Signal sent to Telegram (Local Bridge will execute)")
                reason_snippet = (result.get('reasoning') or result.get('pre_news_bias') or 'No clear directional bias.').strip()
                add_notification('success', f"✅ **{symbol}**: New {result.get('signal')} signal accepted. Score: {combined_score}/100. Entry: {result.get('entry')} | SL: {result.get('stop_loss')} | TP: {result.get('take_profit', ['N/A'])[0] if result.get('take_profit') else 'N/A'}. Reason: {reason_snippet}", symbol=symbol, signal=result.get('signal'), score=combined_score)
                if not is_auto: st.markdown("---")
            else:
                st.session_state.active_signals[symbol] = {'direction': result.get('signal'), 'entry': result.get('entry', 0), 'timestamp': current_time, 'score': combined_score}
                add_notification('info', f"🧭 **{symbol}**: Candidate setup detected with score {combined_score}/100; awaiting confirmation.", symbol=symbol, signal=result.get('signal'), score=combined_score)
        else:
            ai_reason = result.get('display_reasoning') or result.get('reasoning') or result.get('rejection_reason', 'Low confidence or DXY contradiction')
            msg = f"⚪ **{symbol}**: Signal Rejected. Score: {result.get('confluence_score', 0)}/100, Confidence: {result.get('confidence', 'N/A')}. AI Reason: {ai_reason}"
            if not is_auto: st.info(msg)
            add_notification('warning', msg, symbol=symbol, signal=result.get('signal'), score=result.get('confluence_score'))
    news_events = get_high_impact_news(selected_symbols=selected_symbols, reference_dt=datetime.now(timezone.utc))
    sync_news_event_statuses(news_events, selected_symbols=selected_symbols)
    refresh_col, spacer = st.columns([2, 8])
    with refresh_col:
        if st.button("🔄 Refresh Market Data", disabled=st.session_state.bot_running):
            load_market_data(force_refresh=True)
    status_placeholder = st.empty()
    render_analysis_status(status_placeholder)
    def _local_fallback(symbol, all_data):
        m10_local = all_data.get(symbol, {}).get('M10', pd.DataFrame())
        if not m10_local.empty:
            swings_local = find_swings(m10_local)
            pair_config_local = get_pair_config(symbol)
            dxy_data_local = all_data.get('DXY', {}).get('H1', pd.DataFrame())
            dxy_context_local = None
            if not dxy_data_local.empty:
                dxy_micro_local = calculate_microstructure(dxy_data_local)
                dxy_context_local = {'trend': dxy_micro_local['momentum'], 'price_vs_vwap': dxy_micro_local['price_vs_vwap']}
            h4_local = all_data.get(symbol, {}).get('H4', pd.DataFrame())
            htf_local = None
            if not h4_local.empty:
                h4_micro_local = calculate_microstructure(h4_local)
                htf_local = {'trend': h4_micro_local['momentum'], 'bias': h4_micro_local['momentum']}
            pic_local = build_mtf_picture(all_data, symbol)
            firm_local, firm_notes_local = resolve_firm_direction(symbol, pic_local)
            learn_local = learning_note(symbol)
            return build_pre_news_fallback_analysis(symbol, m10_local, swings_local, pair_config_local, dxy_context_local, {'within_2h': False}, htf_context=htf_local, picture=pic_local, firm=firm_local, firm_notes=firm_notes_local, learning=learn_local)
        return {'signal': 'WAIT', 'confidence': 'LOW', 'confluence_score': MINIMUM_CONFLUENCE_SCORE, 'rejection_reason': 'Data unavailable for fallback'}
    if st.button("🔍 Run Macro Analysis Now", type="secondary", disabled=st.session_state.bot_running):
        try:
            if is_gpt_rate_limited():
                st.warning("⏳ Groq rate limit in effect. Please wait before running another analysis.")
            else:
                progress_bar = st.progress(0)
                st.session_state.rate_limit_hit = False
                all_data = load_market_data(force_refresh=False)
                for i, symbol in enumerate(selected_symbols):
                    next_pair = selected_symbols[i + 1] if i + 1 < len(selected_symbols) else None
                    if st.session_state.get('rate_limit_hit', False) or is_gpt_rate_limited():
                        st.warning("⏳ Rate limit reached for GPT. Attempting local fallback or cached result for this symbol.")
                        cached_result = get_cached_analysis(symbol)
                        result = cached_result if cached_result is not None else _local_fallback(symbol, all_data)
                        set_cached_analysis(symbol, result)
                        process_symbol_result(result, symbol, is_auto=False, all_data=all_data)
                    else:
                        update_analysis_status(symbol=symbol, message=f"Starting analysis for {symbol}", next_pair=next_pair)
                        render_analysis_status(status_placeholder)
                        with st.spinner(f"Analyzing {symbol} with DXY Correlation..."):
                            result = analyze_symbol_premium(symbol, all_data, news_override=None)
                        set_cached_analysis(symbol, result)
                        process_symbol_result(result, symbol, is_auto=False, all_data=all_data)
                    if i < len(selected_symbols) - 1:
                        time.sleep(GROQ_MIN_REQUEST_INTERVAL)
                    progress_bar.progress((i + 1) / len(selected_symbols))
                progress_bar.empty()
                st.session_state.last_analysis_time = datetime.now()
                st.session_state.next_check_time = datetime.now() + timedelta(minutes=ANALYSIS_INTERVAL_MINUTES)
                add_notification('info', f"✅ Analysis run complete. Next run at {st.session_state.next_check_time.strftime('%H:%M:%S')}.")
        except Exception as e:
            add_notification('warning', f"Analysis run failed: {str(e)}")
            st.error(f"Analysis run failed: {str(e)}")
    if st.session_state.bot_running:
        if st.session_state.analysis_in_progress:
            st.info("🔄 Analysis is already in progress. Refreshing status and countdown...")
        elif st.session_state.next_check_time:
            time_left = (st.session_state.next_check_time - datetime.now()).total_seconds()
            if time_left > 0:
                st.warning(f"⏳ Waiting {int(time_left)}s until the next scheduled analysis. Keep this page open.")
            elif is_scheduled_run_due():
                scheduled_start = st.session_state.next_check_time or datetime.now()
                if is_gpt_rate_limited():
                    st.warning("⏳ Groq rate limit in effect - using cached/fallback analysis this cycle to keep the strict 5-minute cadence.")
                st.session_state.analysis_in_progress = True
                try:
                    st.info("🔄 Running scheduled analysis...")
                    progress_bar = st.progress(0)
                    st.session_state.rate_limit_hit = False
                    all_data = load_market_data(force_refresh=False)
                    for i, symbol in enumerate(selected_symbols):
                        next_pair = selected_symbols[i + 1] if i + 1 < len(selected_symbols) else None
                        if st.session_state.get('rate_limit_hit', False) or is_gpt_rate_limited():
                            st.warning("⏳ Rate limit reached. Using cached/fallback analysis for this symbol.")
                            cached_result = get_cached_analysis(symbol)
                            result = cached_result if cached_result is not None else _local_fallback(symbol, all_data)
                            update_analysis_status(symbol=symbol, message=f"Rate-limited: cached/fallback analysis for {symbol}", next_pair=next_pair)
                            render_analysis_status(status_placeholder)
                            process_symbol_result(result, symbol, is_auto=True, all_data=all_data)
                        else:
                            update_analysis_status(symbol=symbol, message=f"Starting scheduled analysis for {symbol}", next_pair=next_pair)
                            render_analysis_status(status_placeholder)
                            result = analyze_symbol_premium(symbol, all_data, news_override=None)
                            set_cached_analysis(symbol, result)
                            process_symbol_result(result, symbol, is_auto=True, all_data=all_data)
                        if i < len(selected_symbols) - 1 and not (st.session_state.get('rate_limit_hit', False) or is_gpt_rate_limited()):
                            time.sleep(GROQ_MIN_REQUEST_INTERVAL)
                        progress_bar.progress((i + 1) / len(selected_symbols))
                    progress_bar.empty()
                    now_utc = datetime.now(timezone.utc)
                    fresh_news = get_high_impact_news(selected_symbols=selected_symbols, reference_dt=now_utc)
                    sync_news_event_statuses(fresh_news, selected_symbols=selected_symbols)
                    target_event = pick_news_event_for_analysis(fresh_news, now_utc)
                    if target_event and not (st.session_state.get('rate_limit_hit', False) or is_gpt_rate_limited()):
                        st.info(f"📰 Running pre-news impact analysis for: {target_event.get('event')} ({target_event.get('time')})...")
                        run_news_analysis_cycle(target_event, all_data, selected_symbols)
                    st.session_state.last_analysis_time = datetime.now()
                    st.session_state.next_check_time = scheduled_start + timedelta(minutes=ANALYSIS_INTERVAL_MINUTES)
                    add_notification('info', f"✅ Scheduled analysis complete. Next run at {st.session_state.next_check_time.strftime('%H:%M:%S')}.")
                finally:
                    st.session_state.analysis_in_progress = False
with tab2:
    st.header("📜 Premium Signal History")
    if len(st.session_state.signal_history) == 0:
        st.info("📭 No high-quality signals generated yet.")
    else:
        premium_signals = [s for s in st.session_state.signal_history if s.get('confidence') == 'HIGH' and s.get('confluence_score', 0) >= 80]
        st.metric("Total Premium Signals Logged", len(premium_signals))
        for i, signal in enumerate(reversed(premium_signals)):
            with st.expander(f"{'🟢' if signal.get('signal') == 'BUY' else '🔴'} {signal.get('symbol', 'N/A')} - {signal.get('signal')} | Score: {signal.get('confluence_score')}/100 | {signal.get('timestamp', 'N/A')}", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Entry", signal.get('entry', 'N/A'))
                col_b.metric("Stop Loss", signal.get('stop_loss', 'N/A'))
                col_c.metric("Take Profit", signal.get('take_profit', ['N/A'])[0] if signal.get('take_profit') else 'N/A')
                st.write(f"**Bias:** {signal.get('bias')} | **Confidence:** {signal.get('confidence')}")
                st.write(f"**DXY Correlation:** {signal.get('dxy_correlation', 'N/A')}")
                st.write(f"**Reasoning:** {signal.get('reasoning')}")
                st.markdown("---")
with tab3:
    st.header("🔔 Analysis Results & Notifications")
    st.markdown("Live-market analysis results and system events. (Pre-news impact signals are delivered in the News Calendar tab and Telegram only.)")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 2, 1, 1])
    with ctrl_col1:
        filter_type = st.selectbox("Filter by type", ["All", "Signals Only", "Warnings", "Info", "Success"], key="notif_filter")
    with ctrl_col2:
        filter_symbol = st.selectbox("Filter by symbol", ["All"] + selected_symbols, key="notif_symbol_filter")
    with ctrl_col3:
        if st.button("✅ Mark All Read", use_container_width=True):
            mark_all_notifications_read()
            st.rerun()
    with ctrl_col4:
        if st.button("🗑️ Clear All", use_container_width=True):
            clear_notifications()
            st.rerun()
    unread = st.session_state.get('unread_count', 0)
    if unread > 0:
        st.info(f"📬 You have **{unread}** unread notification(s).")
    notifications = get_notifications()
    filtered_notifications = list(reversed(notifications))
    if filter_type == "Signals Only":
        filtered_notifications = [n for n in filtered_notifications if n.get('signal') in ('BUY', 'SELL')]
    elif filter_type == "Warnings":
        filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'warning']
    elif filter_type == "Info":
        filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'info']
    elif filter_type == "Success":
        filtered_notifications = [n for n in filtered_notifications if n.get('type') == 'success']
    if filter_symbol != "All":
        filtered_notifications = [n for n in filtered_notifications if n.get('symbol') == filter_symbol]
    if not filtered_notifications:
        st.info("📭 No notifications match the current filter. Start the bot and run an analysis to see results here.")
    else:
        st.caption(f"Showing {len(filtered_notifications)} notification(s)")
        for note in filtered_notifications:
            note_type = note.get('type', 'info')
            badges = []
            if note.get('symbol'):
                badges.append(f"`{note['symbol']}`")
            if note.get('signal'):
                sig_emoji = "🟢" if note['signal'] == "BUY" else "🔴" if note['signal'] == "SELL" else "⚪"
                badges.append(f"{sig_emoji} {note['signal']}")
            if note.get('score') is not None:
                badges.append(f"📈 {note['score']}/100")
            header = f"**[{note.get('time', '')}]** {' '.join(badges)}" if badges else f"**[{note.get('time', '')}]**"
            if note_type == 'success':
                st.success(f"{header}\n{note.get('message', '')}")
            elif note_type == 'warning':
                st.warning(f"{header}\n{note.get('message', '')}")
            else:
                st.info(f"{header}\n{note.get('message', '')}")
with tab4:
    st.header("📰 High-Impact News Calendar & Pre-News Impact Signals")
    st.caption(f"Pre-news impact signals are analysed once per event and dispatched at least {NEWS_PRE_WINDOW_HOURS} hours before the release — to this tab and Telegram only. Signals include direction and reasoning only (no Entry/SL/TP).")
    if st.button("🔄 Refresh News"): st.rerun()
    news = get_high_impact_news(selected_symbols=selected_symbols, reference_dt=datetime.now(timezone.utc))
    sync_news_event_statuses(news, selected_symbols=selected_symbols)
    if news:
        for n in news:
            event_id = n.get('event_id') or f"{n.get('event')}|{n.get('currency')}|{n.get('time')}"
            status_meta = st.session_state.news_event_statuses.get(event_id, {})
            status = status_meta.get('status', 'waiting')
            detail = status_meta.get('detail', 'Waiting for AI pre-news analysis.')
            urgency = "🟠 Within 2 hours" if n.get('within_2h') else "🟡 Upcoming"
            st.markdown(f"**{n['time']}** - {urgency} - {n['currency']}: {n['event']}")
            if status == 'sent':
                st.success(f"✅ Pre-news signal sent once • {detail}")
            elif status == 'analyzing':
                st.info(f"🧠 AI pre-news analysis in progress • {detail}")
            elif status == 'window_closed':
                st.warning(f"⛔ {detail}")
            elif status == 'expired':
                st.caption(f"⌛ {detail}")
            else:
                st.caption(f"⏳ {detail}")
            stored = st.session_state.news_event_results.get(event_id)
            if stored:
                for symbol, a in stored['results'].items():
                    sig = a.get('signal', 'SKIPPED')
                    if sig in ('BUY', 'SELL'):
                        sig_emoji = "🟢" if sig == 'BUY' else "🔴"
                        st.success(f"{sig_emoji} **{symbol}**: {sig}")
                        # Show historical pattern if available
                        hist_pattern = a.get('historical_pattern', '')
                        if hist_pattern:
                            st.caption(f"📚 Historical: {hist_pattern[:300]}")
                        st.caption((a.get('reasoning') or '')[:500])
                    elif sig == 'WAIT':
                        st.info(f"⚪ **{symbol}**: WAIT — {(a.get('rejection_reason') or a.get('reasoning') or '')[:300]}")
                    else:
                        st.warning(f"⚪ **{symbol}**: skipped ({a.get('reason', 'rate limit')}) — will retry next cycle if window still open.")
            st.markdown("---")
    else:
        st.info("No high-impact news in the upcoming window")
with tab5:
    st.header("⚙️ System Settings")
    st.subheader("📱 Telegram & Local Bridge Setup")
    st.markdown("""
    1. Create a bot via @BotFather on Telegram and get your chat ID.
    2. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to Streamlit Secrets.
    3. **For Auto-Execution:** Run the `local_mt5_bridge.py` script on your Windows PC.
       It will listen to this Telegram channel and execute trades directly into your MT5 terminal.
    """)
    st.subheader("🎯 Quality Filters")
    st.info(f"""**Current Active Settings:**
- AI Model: **llama-3.3-70b-versatile** (falls back to llama-3.1-8b-instant)
- Minimum Confluence Score: **{MINIMUM_CONFLUENCE_SCORE}/100**
- **Firm Desk Bias with Hysteresis:** HTF-first standing bias ({BIAS_MIN_HOLD_MINUTES}-min hold) prevents BUY↔SELL flip-flopping; flips require a real H1/H4 structural break with strong evidence.
- **Cooldown:** a good setup is not re-signalled for **30 minutes**.
- **Adaptive Learning:** every accepted signal is scored against what price actually does; hit-rate tunes conviction.
- **Rich Reasoning Enforcement:** shallow AI reasoning is expanded into a full institutional brief; fallback uses the MTF desk picture.
- **Pre-News Impact Engine:** each high-impact event analysed ONCE (>= {NEWS_PRE_WINDOW_HOURS}h ahead), sent to News Calendar + Telegram only. Signals include direction and reasoning only (no Entry/SL/TP).
- **Gold Entry Rule:** XAUUSD entries hard-clamped within 10 points of the live price.
- **Strict 5-Minute Cadence** anchored to scheduled start; rate limits never extend the interval.""")
# ── Auto-Refresh Loop (MUST be the LAST block in the script) ─────────────────
if st.session_state.bot_running and not st.session_state.analysis_in_progress:
    time.sleep(10)
    st.rerun()
