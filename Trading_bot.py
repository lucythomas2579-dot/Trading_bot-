import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import pickle
from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

# ============================================
# PAGE CONFIG - MOBILE FRIENDLY
# ============================================
st.set_page_config(
    page_title="Trading Bot",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Mobile friendly CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .stButton button {
        width: 100%;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .stMetric {
        font-size: 0.9rem;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.8rem;
        font-size: 0.8rem;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    h2 {
        font-size: 1.3rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ENUMS & DATA CLASSES
# ============================================
class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"

class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

@dataclass
class MarketStructure:
    trend: Trend
    trend_strength: float
    bos: List[Dict]
    choch: List[Dict]
    swing_highs: List[float]
    swing_lows: List[float]
    resistance: List[float]
    support: List[float]
    premium_zone: float
    discount_zone: float

@dataclass
class TradeDecision:
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: float
    position_size: float
    reasons: List[str]
    market_structure: MarketStructure
    ml_probability: float
    signal_strength: SignalStrength
    filter_checks: Dict[str, bool]

# ============================================
# INDICATOR CALCULATOR
# ============================================
def calculate_indicators(df):
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    try:
        # Moving Averages
        df['MA9'] = df['close'].rolling(9).mean()
        df['MA21'] = df['close'].rolling(21).mean()
        df['MA50'] = df['close'].rolling(50).mean()
        df['MA200'] = df['close'].rolling(200).mean()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_low'] = bb.bollinger_lband()
        
        # ATR
        df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        
        # MFI
        df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()
        
        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum().replace(0, 1)
        
        # Clean up
        df = df.fillna(method='ffill').fillna(method='bfill')
        
    except Exception as e:
        st.error(f"Indicator error: {str(e)}")
    
    return df

# ============================================
# MARKET STRUCTURE ANALYZER
# ============================================
def analyze_structure(df):
    if df is None or df.empty:
        return MarketStructure(Trend.RANGING, 0, [], [], [], [], [], [], 0, 0)
    
    # Ensure indicators are calculated
    if 'MA50' not in df.columns:
        df = calculate_indicators(df)
    
    swing_highs = []
    swing_lows = []
    bos = []
    choch = []
    
    for i in range(2, len(df) - 2):
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i+2]):
            swing_highs.append(df['high'].iloc[i])
        
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            swing_lows.append(df['low'].iloc[i])
    
    # Detect BOS
    for i in range(1, len(swing_highs)):
        if swing_highs[i] > swing_highs[i-1]:
            bos.append({'type': 'bullish', 'price': swing_highs[i]})
    
    for i in range(1, len(swing_lows)):
        if swing_lows[i] < swing_lows[i-1]:
            bos.append({'type': 'bearish', 'price': swing_lows[i]})
    
    # Detect CHoCH
    for i in range(1, len(bos)):
        if bos[i]['type'] != bos[i-1]['type']:
            choch.append({'from': bos[i-1]['type'], 'to': bos[i]['type'], 'price': bos[i]['price']})
    
    # Determine trend
    if len(bos) > 2:
        recent_bos = bos[-5:]
        bullish_count = sum(1 for b in recent_bos if b['type'] == 'bullish')
        bearish_count = sum(1 for b in recent_bos if b['type'] == 'bearish')
        
        if bullish_count > bearish_count + 2:
            trend = Trend.BULLISH
            trend_strength = min(100, bullish_count * 20)
        elif bearish_count > bullish_count + 2:
            trend = Trend.BEARISH
            trend_strength = min(100, bearish_count * 20)
        else:
            trend = Trend.RANGING
            trend_strength = 50
    else:
        trend = Trend.RANGING
        trend_strength = 50
    
    # Support and Resistance
    resistance = sorted(swing_highs[-5:], reverse=True)[:3] if swing_highs else []
    support = sorted(swing_lows[-5:])[:3] if swing_lows else []
    
    # Premium/Discount zones
    recent_range = df.tail(50)
    range_high = recent_range['high'].max()
    range_low = recent_range['low'].min()
    range_size = range_high - range_low
    
    premium_zone = range_low + range_size * 0.618
    discount_zone = range_low + range_size * 0.382
    
    return MarketStructure(
        trend=trend,
        trend_strength=trend_strength,
        bos=bos,
        choch=choch,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        resistance=resistance,
        support=support,
        premium_zone=premium_zone,
        discount_zone=discount_zone
    )

# ============================================
# DATA FETCHER
# ============================================
@st.cache_data(ttl=300)
def fetch_data(symbol, timeframe='15m', limit=200):
    try:
        yahoo_interval = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '60m', '4h': '1h', '1d': '1d', '1w': '1wk'
        }.get(timeframe, '5m')
        
        if '/' in symbol:
            ticker = symbol.replace('/', '') + '=X'
        else:
            ticker = symbol
        
        df = yf.download(ticker, period='5d', interval=yahoo_interval, progress=False)
        
        if df.empty:
            return None
        
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['Datetime'])
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(limit)
        
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

# ============================================
# SIGNAL GENERATOR
# ============================================
def generate_signal(df_short, df_medium, df_long, ml_prob=0.5):
    if any(df is None for df in [df_short, df_medium, df_long]):
        return None
    
    structure_short = analyze_structure(df_short)
    structure_medium = analyze_structure(df_medium)
    structure_long = analyze_structure(df_long)
    
    current_price = df_short['close'].iloc[-1]
    rsi = df_short['RSI'].iloc[-1]
    macd = df_short['MACD'].iloc[-1]
    macd_signal = df_short['MACD_signal'].iloc[-1]
    adx = df_short['ADX'].iloc[-1]
    
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # 1. Higher Timeframe Trend
    if structure_long.trend == Trend.BULLISH:
        buy_score += 30
        reasons.append("✅ Long-term bullish")
    elif structure_long.trend == Trend.BEARISH:
        sell_score += 30
        reasons.append("✅ Long-term bearish")
    
    # 2. Medium Trend
    if structure_medium.trend == Trend.BULLISH:
        buy_score += 20
        reasons.append("✅ Medium-term bullish")
    elif structure_medium.trend == Trend.BEARISH:
        sell_score += 20
        reasons.append("✅ Medium-term bearish")
    
    # 3. Short Trend
    if structure_short.trend == Trend.BULLISH:
        buy_score += 15
        reasons.append("✅ Short-term bullish")
    elif structure_short.trend == Trend.BEARISH:
        sell_score += 15
        reasons.append("✅ Short-term bearish")
    
    # 4. BOS
    if structure_short.bos:
        last_bos = structure_short.bos[-1]
        if last_bos['type'] == 'bullish':
            buy_score += 15
            reasons.append("✅ Bullish BOS")
        else:
            sell_score += 15
            reasons.append("✅ Bearish BOS")
    
    # 5. Premium/Discount
    if current_price <= structure_short.discount_zone:
        buy_score += 15
        reasons.append("✅ In discount zone")
    elif current_price >= structure_short.premium_zone:
        sell_score += 15
        reasons.append("✅ In premium zone")
    
    # 6. RSI
    if rsi < 30:
        buy_score += 10
        reasons.append(f"✅ RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        sell_score += 10
        reasons.append(f"✅ RSI overbought ({rsi:.1f})")
    
    # 7. MACD
    if macd > macd_signal:
        buy_score += 8
        reasons.append("✅ MACD bullish")
    else:
        sell_score += 8
        reasons.append("✅ MACD bearish")
    
    # 8. ADX
    if adx > 25:
        if buy_score > sell_score:
            buy_score += 10
            reasons.append(f"✅ Strong trend (ADX: {adx:.1f})")
        else:
            sell_score += 10
            reasons.append(f"✅ Strong trend (ADX: {adx:.1f})")
    
    # 9. ML Probability
    if ml_prob > 0.6:
        buy_score += ml_prob * 20
        reasons.append(f"✅ ML bullish ({ml_prob:.0%})")
    elif ml_prob < 0.4:
        sell_score += (1 - ml_prob) * 20
        reasons.append(f"✅ ML bearish ({1-ml_prob:.0%})")
    
    # Determine Signal
    total_score = max(buy_score, sell_score)
    
    if total_score >= 80:
        signal_strength = SignalStrength.VERY_STRONG
    elif total_score >= 60:
        signal_strength = SignalStrength.STRONG
    elif total_score >= 40:
        signal_strength = SignalStrength.MODERATE
    else:
        signal_strength = SignalStrength.WEAK
    
    if buy_score > sell_score + 25 and buy_score >= 50:
        action = 'BUY'
        confidence = min(100, buy_score)
    elif sell_score > buy_score + 25 and sell_score >= 50:
        action = 'SELL'
        confidence = min(100, sell_score)
    else:
        action = 'HOLD'
        confidence = max(buy_score, sell_score) / 2
        if not reasons:
            reasons = ["⚠️ No clear confluence"]
    
    reasons.append(f"📊 Buy: {buy_score:.0f} | Sell: {sell_score:.0f}")
    
    # Calculate SL and TP
    atr = df_short['ATR'].iloc[-1]
    
    if action == 'BUY':
        if structure_short.support:
            sl = min(structure_short.support) * 0.995
        else:
            sl = current_price - atr * 2
        
        tp1 = current_price + (current_price - sl) * 1.5
        tp2 = current_price + (current_price - sl) * 2.5
        
        if structure_short.resistance:
            next_res = min([r for r in structure_short.resistance if r > current_price], default=None)
            if next_res:
                tp1 = min(tp1, next_res * 0.995)
    
    elif action == 'SELL':
        if structure_short.resistance:
            sl = max(structure_short.resistance) * 1.005
        else:
            sl = current_price + atr * 2
        
        tp1 = current_price - (sl - current_price) * 1.5
        tp2 = current_price - (sl - current_price) * 2.5
        
        if structure_short.support:
            next_sup = max([s for s in structure_short.support if s < current_price], default=None)
            if next_sup:
                tp1 = max(tp1, next_sup * 1.005)
    
    else:
        sl = 0
        tp1 = 0
        tp2 = 0
    
    risk = abs(current_price - sl) if sl > 0 else 0
    reward = abs(tp1 - current_price) if tp1 > 0 else 0
    risk_reward = reward / risk if risk > 0 else 0
    
    filter_checks = {
        'rr_ok': risk_reward >= 1.5,
        'confidence_ok': confidence >= 60,
        'trend_ok': structure_short.trend != Trend.RANGING
    }
    
    return TradeDecision(
        action=action,
        confidence=confidence,
        entry_price=current_price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_reward=risk_reward,
        position_size=0,
        reasons=reasons,
        market_structure=structure_short,
        ml_probability=ml_prob,
        signal_strength=signal_strength,
        filter_checks=filter_checks
    )

# ============================================
# ML PREDICTOR (Simplified)
# ============================================
@st.cache_resource
def get_ml_predictor():
    return None  # Simplified version

# ============================================
# UI
# ============================================
st.title("📊 Trading Bot")
st.caption("Professional-grade analysis on your phone")

# Main container
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        asset_type = st.selectbox(
            "Asset",
            ["EUR/USD", "GBP/USD", "BTC/USDT", "ETH/USDT", "AAPL", "GOOGL"]
        )
    
    with col2:
        timeframe = st.selectbox(
            "TF",
            ["15m", "1h", "4h", "1d"]
        )

# Risk settings
with st.expander("⚙️ Risk Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        balance = st.number_input("Balance ($)", min_value=100, value=10000, step=100)
    with col2:
        risk_percent = st.slider("Risk %", 0.5, 5.0, 2.0, 0.5)

# Analyze button
if st.button("🔍 ANALYZE", type="primary"):
    with st.spinner("Analyzing market..."):
        # Fetch data for all timeframes
        tf_map = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
        
        short_tf = tf_map.get(timeframe, '15m')
        medium_tf = '1h' if short_tf != '1h' else '4h'
        long_tf = '4h' if medium_tf != '4h' else '1d'
        
        df_short = fetch_data(asset_type, short_tf, 100)
        df_medium = fetch_data(asset_type, medium_tf, 100)
        df_long = fetch_data(asset_type, long_tf, 100)
        
        if all(df is not None for df in [df_short, df_medium, df_long]):
            # Calculate indicators
            df_short = calculate_indicators(df_short)
            df_medium = calculate_indicators(df_medium)
            df_long = calculate_indicators(df_long)
            
            # Generate signal
            decision = generate_signal(df_short, df_medium, df_long)
            
            if decision:
                # Display Signal
                if decision.action == 'BUY':
                    st.success(f"🚀 BUY SIGNAL")
                elif decision.action == 'SELL':
                    st.error(f"🔻 SELL SIGNAL")
                else:
                    st.warning(f"⏳ HOLD")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{decision.confidence:.0f}%")
                with col2:
                    st.metric("Entry", f"${decision.entry_price:.4f}")
                with col3:
                    st.metric("R:R", f"{decision.risk_reward:.2f}")
                
                # SL and TP
                st.subheader("🎯 Levels")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stop Loss", f"${decision.stop_loss:.4f}" if decision.stop_loss > 0 else "N/A")
                with col2:
                    st.metric("TP1", f"${decision.take_profit_1:.4f}" if decision.take_profit_1 > 0 else "N/A")
                with col3:
                    st.metric("TP2", f"${decision.take_profit_2:.4f}" if decision.take_profit_2 > 0 else "N/A")
                
                # Trend
                st.subheader("📊 Trend")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Trend", decision.market_structure.trend.value)
                with col2:
                    st.metric("Strength", f"{decision.market_structure.trend_strength:.0f}%")
                
                # Filter Checks
                if decision.filter_checks:
                    st.subheader("🔍 Filters")
                    cols = st.columns(len(decision.filter_checks))
                    for i, (key, value) in enumerate(decision.filter_checks.items()):
                        with cols[i]:
                            st.metric(key.replace('_', ' ').title(), "✅" if value else "❌")
                
                # Reasons
                with st.expander("📝 Reasons", expanded=True):
                    for reason in decision.reasons:
                        st.write(reason)
                
                # Position Size
                if decision.action != 'HOLD':
                    risk_amount = balance * (risk_percent / 100)
                    if decision.stop_loss > 0:
                        risk_per_share = abs(decision.entry_price - decision.stop_loss)
                        position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
                        position_value = position_size * decision.entry_price
                        
                        st.subheader("💰 Position Size")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Amount", f"${risk_amount:.2f}")
                        with col2:
                            st.metric("Position Size", f"{position_size:.4f}")
                        with col3:
                            st.metric("Position Value", f"${position_value:.2f}")
                
                # Chart
                st.subheader("📈 Price Chart")
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_short['timestamp'].tail(50),
                    open=df_short['open'].tail(50),
                    high=df_short['high'].tail(50),
                    low=df_short['low'].tail(50),
                    close=df_short['close'].tail(50)
                ))
                
                if 'MA9' in df_short.columns:
                    fig.add_trace(go.Scatter(
                        x=df_short['timestamp'].tail(50),
                        y=df_short['MA9'].tail(50),
                        name='MA9',
                        line=dict(color='orange', width=1)
                    ))
                
                if 'MA50' in df_short.columns:
                    fig.add_trace(go.Scatter(
                        x=df_short['timestamp'].tail(50),
                        y=df_short['MA50'].tail(50),
                        name='MA50',
                        line=dict(color='blue', width=1)
                    ))
                
                if decision.action != 'HOLD':
                    if decision.stop_loss > 0:
                        fig.add_hline(y=decision.stop_loss, line_dash="dash", line_color="red",
                                     annotation_text="SL", annotation_position="bottom right")
                    if decision.take_profit_1 > 0:
                        fig.add_hline(y=decision.take_profit_1, line_dash="dash", line_color="green",
                                     annotation_text="TP1", annotation_position="top right")
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="",
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Timestamp
                st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("❌ Failed to fetch data")

# Footer
st.markdown("---")
st.caption("⚠️ Not financial advice. For educational purposes only.")
