import numpy as np
import pandas as pd

ARIMA_ORDER   = (1, 0, 1)
ARIMA_WINDOW  = 120
MA_SHORT      = 20
MA_LONG       = 60
GARCH_WINDOW  = 100
VOL_THRESHOLD = 2.0


def _arima_signals(close):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return pd.Series(0, index=close.index)

    ret     = close.pct_change()
    signals = pd.Series(0, index=close.index)

    for t in range(ARIMA_WINDOW, len(ret) - 1):
        window = ret.iloc[t - ARIMA_WINDOW:t].dropna().values
        if len(window) < 30:
            continue
        try:
            fc = float(ARIMA(window, order=ARIMA_ORDER).fit().forecast(steps=1)[0])
            signals.iloc[t + 1] = 1 if fc > 0 else -1
        except Exception:
            pass

    return signals


def _garch_vol_filter(close):
    try:
        from arch import arch_model
    except ImportError:
        ret = close.pct_change() * 100
        vol = ret.rolling(20).std()
        return (vol / vol.rolling(GARCH_WINDOW).mean().replace(0, np.nan)) <= VOL_THRESHOLD

    ret_pct = close.pct_change().dropna() * 100
    ok_mask = pd.Series(True, index=close.index)
    try:
        model    = arch_model(ret_pct, vol='Garch', p=1, q=1, dist='t').fit(disp='off')
        cond_vol = pd.Series(np.sqrt(model.conditional_volatility.values), index=ret_pct.index)
        vol_ratio = cond_vol / cond_vol.rolling(GARCH_WINDOW).mean().replace(0, np.nan)
        for idx in ok_mask.index:
            if idx in vol_ratio.index:
                ok_mask.loc[idx] = vol_ratio.loc[idx] <= VOL_THRESHOLD
    except Exception:
        pass

    return ok_mask


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]

    df["MA_short"]  = close.rolling(MA_SHORT).mean()
    df["MA_long"]   = close.rolling(MA_LONG).mean()
    trend           = np.where(df["MA_short"] > df["MA_long"], 1, -1)

    df["arima_sig"] = _arima_signals(close)
    df["vol_ok"]    = _garch_vol_filter(close)

    arima  = df["arima_sig"].values
    vol_ok = df["vol_ok"].values

    df["position"] = np.where(
        vol_ok & (arima == 1)  & (trend == 1),   1,
        np.where(
        vol_ok & (arima == -1) & (trend == -1), -1, 0)
    )
    df["position"] = df["position"].fillna(0).astype(int)

    return df
