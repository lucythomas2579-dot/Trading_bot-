import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Trading Bot", layout="centered")
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; }
    .stButton button { width: 100%; }
    h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_data(symbol, interval='15m'):
    try:
        if '/' in symbol:
            ticker = symbol.replace('/', '') + '=X'
        else:
            ticker = symbol
        
        df = yf.download(ticker, period='5d', interval=interval, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        return df[['open', 'high', 'low', 'close', 'volume']].tail(100)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_signal(df):
    if df is None or df.empty:
        return None
    
    df = df.copy()
    df['MA9'] = df['close'].rolling(9).mean()
    df['MA21'] = df['close'].rolling(21).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    
    current = df.iloc[-1]
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # Trend
    if current['close'] > current['MA50']:
        buy_score += 20
        reasons.append("✅ Price above 50 MA (uptrend)")
    else:
        sell_score += 20
        reasons.append("✅ Price below 50 MA (downtrend)")
    
    # MA Crossover
    if current['MA9'] > current['MA21']:
        buy_score += 15
        reasons.append("✅ Bullish MA crossover")
    else:
        sell_score += 15
        reasons.append("✅ Bearish MA crossover")
    
    # RSI
    rsi = current['RSI']
    if rsi < 30:
        buy_score += 15
        reasons.append(f"✅ RSI oversold ({rsi:.1f})")
    elif rsi > 70:
        sell_score += 15
        reasons.append(f"✅ RSI overbought ({rsi:.1f})")
    else:
        reasons.append(f"ℹ️ RSI neutral ({rsi:.1f})")
    
    # Support/Resistance
    if current['close'] <= recent_low * 1.01:
        buy_score += 10
        reasons.append("✅ Near support level")
    elif current['close'] >= recent_high * 0.99:
        sell_score += 10
        reasons.append("✅ Near resistance level")
    
    # Signal
    if buy_score > sell_score + 20 and buy_score >= 40:
        signal = "BUY"
        confidence = min(100, buy_score)
        atr = current['ATR']
        sl = recent_low * 0.98 if current['close'] <= recent_low * 1.01 else current['close'] - atr * 2
        tp1 = current['close'] + (current['close'] - sl) * 1.5
        tp2 = current['close'] + (current['close'] - sl) * 2.5
    elif sell_score > buy_score + 20 and sell_score >= 40:
        signal = "SELL"
        confidence = min(100, sell_score)
        atr = current['ATR']
        sl = recent_high * 1.02 if current['close'] >= recent_high * 0.99 else current['close'] + atr * 2
        tp1 = current['close'] - (sl - current['close']) * 1.5
        tp2 = current['close'] - (sl - current['close']) * 2.5
    else:
        signal = "HOLD"
        confidence = max(buy_score, sell_score) / 2
        sl = 0
        tp1 = 0
        tp2 = 0
        reasons.append("⏳ No clear signal")
    
    reasons.append(f"📊 Buy Score: {buy_score:.0f} | Sell Score: {sell_score:.0f}")
    
    risk = abs(current['close'] - sl) if sl > 0 else 0
    reward = abs(tp1 - current['close']) if tp1 > 0 else 0
    rr = reward / risk if risk > 0 else 0
    
    return {
        'signal': signal,
        'confidence': confidence,
        'price': current['close'],
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'rr': rr,
        'reasons': reasons,
        'df': df
    }

st.title("📊 Trading Bot")

col1, col2 = st.columns(2)
with col1:
    symbol = st.selectbox("Asset", ["EUR/USD", "GBP/USD", "BTC/USDT", "ETH/USDT", "AAPL", "GOOGL"])
with col2:
    interval = st.selectbox("Timeframe", ["15m", "1h", "4h"])

col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Balance ($)", 100, 100000, 10000)
with col2:
    risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0, 0.5)

if st.button("🔍 ANALYZE", type="primary"):
    with st.spinner("Analyzing..."):
        df = fetch_data(symbol, interval)
        if df is not None:
            result = generate_signal(df)
            if result:
                if result['signal'] == 'BUY':
                    st.success(f"🚀 BUY SIGNAL")
                elif result['signal'] == 'SELL':
                    st.error(f"🔻 SELL SIGNAL")
                else:
                    st.warning(f"⏳ HOLD")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence']:.0f}%")
                with col2:
                    st.metric("Price", f"${result['price']:.4f}")
                with col3:
                    st.metric("R:R", f"{result['rr']:.2f}")
                
                if result['signal'] != 'HOLD':
                    st.subheader("🎯 Levels")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Stop Loss", f"${result['sl']:.4f}")
                    with col2:
                        st.metric("TP1", f"${result['tp1']:.4f}")
                    with col3:
                        st.metric("TP2", f"${result['tp2']:.4f}")
                    
                    risk_amount = balance * (risk_pct / 100)
                    if result['sl'] > 0:
                        risk_per_share = abs(result['price'] - result['sl'])
                        position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
                        position_value = position_size * result['price']
                        
                        st.subheader("💰 Position Size")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Amount", f"${risk_amount:.2f}")
                        with col2:
                            st.metric("Position Size", f"{position_size:.4f}")
                        with col3:
                            st.metric("Position Value", f"${position_value:.2f}")
                
                with st.expander("📝 Analysis", expanded=True):
                    for reason in result['reasons']:
                        st.write(reason)
                
                st.subheader("📈 Chart")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                ))
                if result['signal'] != 'HOLD':
                    if result['sl'] > 0:
                        fig.add_hline(y=result['sl'], line_dash="dash", line_color="red", annotation_text="SL")
                    if result['tp1'] > 0:
                        fig.add_hline(y=result['tp1'], line_dash="dash", line_color="green", annotation_text="TP1")
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("❌ Failed to fetch data")

st.markdown("---")
st.caption("⚠️ For educational purposes only")
