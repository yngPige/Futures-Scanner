"""
Custom Technical Indicators Module

This module provides custom implementations of technical indicators
that may have compatibility issues with the original libraries.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def squeeze_pro(high, low, close, bb_length=None, bb_std=None, kc_length=None, kc_scalar_wide=None, kc_scalar_normal=None, kc_scalar_narrow=None, mom_length=None, mom_smooth=None, use_tr=None, mamode=None, offset=None, **kwargs):
    """
    Custom implementation of Squeeze Momentum (SQZ) PRO indicator.
    This is a modified version of the pandas-ta squeeze_pro indicator
    that uses np.nan instead of np.NaN for compatibility with newer numpy versions.
    
    For full documentation, see the original pandas-ta implementation.
    """
    try:
        # Import required functions from pandas_ta
        from pandas_ta.momentum import mom
        from pandas_ta.overlap import ema, sma
        from pandas_ta.trend import decreasing, increasing
        from pandas_ta.volatility import bbands, kc
        from pandas_ta.utils import get_offset
        from pandas_ta.utils import unsigned_differences, verify_series
        
        # Validate arguments
        bb_length = int(bb_length) if bb_length and bb_length > 0 else 20
        bb_std = float(bb_std) if bb_std and bb_std > 0 else 2.0
        kc_length = int(kc_length) if kc_length and kc_length > 0 else 20
        kc_scalar_wide = float(kc_scalar_wide) if kc_scalar_wide and kc_scalar_wide > 0 else 2
        kc_scalar_normal = float(kc_scalar_normal) if kc_scalar_normal and kc_scalar_normal > 0 else 1.5
        kc_scalar_narrow = float(kc_scalar_narrow) if kc_scalar_narrow and kc_scalar_narrow > 0 else 1
        mom_length = int(mom_length) if mom_length and mom_length > 0 else 12
        mom_smooth = int(mom_smooth) if mom_smooth and mom_smooth > 0 else 6

        _length = max(bb_length, kc_length, mom_length, mom_smooth)
        high = verify_series(high, _length)
        low = verify_series(low, _length)
        close = verify_series(close, _length)
        offset = get_offset(offset)

        valid_kc_scaler = kc_scalar_wide > kc_scalar_normal and kc_scalar_normal > kc_scalar_narrow

        if not valid_kc_scaler: return
        if high is None or low is None or close is None: return

        use_tr = kwargs.setdefault("tr", True)
        asint = kwargs.pop("asint", True)
        detailed = kwargs.pop("detailed", False)
        mamode = mamode if isinstance(mamode, str) else "sma"

        def simplify_columns(df, n=3):
            df.columns = df.columns.str.lower()
            return [c.split("_")[0][n - 1:n] for c in df.columns]

        # Calculate Result
        bbd = bbands(close, length=bb_length, std=bb_std, mamode=mamode)
        kch_wide = kc(high, low, close, length=kc_length, scalar=kc_scalar_wide, mamode=mamode, tr=use_tr)
        kch_normal = kc(high, low, close, length=kc_length, scalar=kc_scalar_normal, mamode=mamode, tr=use_tr)
        kch_narrow = kc(high, low, close, length=kc_length, scalar=kc_scalar_narrow, mamode=mamode, tr=use_tr)

        # Simplify KC and BBAND column names for dynamic access
        bbd.columns = simplify_columns(bbd)
        kch_wide.columns = simplify_columns(kch_wide)
        kch_normal.columns = simplify_columns(kch_normal)
        kch_narrow.columns = simplify_columns(kch_narrow)

        momo = mom(close, length=mom_length)
        if mamode.lower() == "ema":
            squeeze = ema(momo, length=mom_smooth)
        else: # "sma"
            squeeze = sma(momo, length=mom_smooth)

        # Classify Squeezes
        squeeze_on_wide = (bbd.l > kch_wide.l) & (bbd.u < kch_wide.u)
        squeeze_on_normal = (bbd.l > kch_normal.l) & (bbd.u < kch_normal.u)
        squeeze_on_narrow = (bbd.l > kch_narrow.l) & (bbd.u < kch_narrow.u)
        squeeze_off_wide = (bbd.l < kch_wide.l) & (bbd.u > kch_wide.u)
        no_squeeze = ~squeeze_on_wide & ~squeeze_off_wide

        # Offset
        if offset != 0:
            squeeze = squeeze.shift(offset)
            squeeze_on_wide = squeeze_on_wide.shift(offset)
            squeeze_on_normal = squeeze_on_normal.shift(offset)
            squeeze_on_narrow = squeeze_on_narrow.shift(offset)
            squeeze_off_wide = squeeze_off_wide.shift(offset)
            no_squeeze = no_squeeze.shift(offset)

        # Handle fills
        if "fillna" in kwargs:
            squeeze.fillna(kwargs["fillna"], inplace=True)
            squeeze_on_wide.fillna(kwargs["fillna"], inplace=True)
            squeeze_on_normal.fillna(kwargs["fillna"], inplace=True)
            squeeze_on_narrow.fillna(kwargs["fillna"], inplace=True)
            squeeze_off_wide.fillna(kwargs["fillna"], inplace=True)
            no_squeeze.fillna(kwargs["fillna"], inplace=True)
        if "fill_method" in kwargs:
            squeeze.fillna(method=kwargs["fill_method"], inplace=True)
            squeeze_on_wide.fillna(method=kwargs["fill_method"], inplace=True)
            squeeze_on_normal.fillna(method=kwargs["fill_method"], inplace=True)
            squeeze_on_narrow.fillna(method=kwargs["fill_method"], inplace=True)
            squeeze_off_wide.fillna(method=kwargs["fill_method"], inplace=True)
            no_squeeze.fillna(method=kwargs["fill_method"], inplace=True)

        # Name and Categorize it
        _props = "" if use_tr else "hlr"
        _props += f"_{bb_length}_{bb_std}_{kc_length}_{kc_scalar_wide}_{kc_scalar_normal}_{kc_scalar_narrow}"
        squeeze.name = f"SQZPRO{_props}"

        data = {
            squeeze.name: squeeze,
            f"SQZPRO_ON_WIDE": squeeze_on_wide.astype(int) if asint else squeeze_on_wide,
            f"SQZPRO_ON_NORMAL": squeeze_on_normal.astype(int) if asint else squeeze_on_normal,
            f"SQZPRO_ON_NARROW": squeeze_on_narrow.astype(int) if asint else squeeze_on_narrow,
            f"SQZPRO_OFF": squeeze_off_wide.astype(int) if asint else squeeze_off_wide,
            f"SQZPRO_NO": no_squeeze.astype(int) if asint else no_squeeze,
        }
        df = DataFrame(data)
        df.name = squeeze.name
        df.category = squeeze.category = "momentum"

        # Detailed Squeeze Series
        if detailed:
            pos_squeeze = squeeze[squeeze >= 0]
            neg_squeeze = squeeze[squeeze < 0]

            pos_inc, pos_dec = unsigned_differences(pos_squeeze, asint=True)
            neg_inc, neg_dec = unsigned_differences(neg_squeeze, asint=True)

            pos_inc *= squeeze
            pos_dec *= squeeze
            neg_dec *= squeeze
            neg_inc *= squeeze

            # Use np.nan instead of npNaN
            pos_inc.replace(0, np.nan, inplace=True)
            pos_dec.replace(0, np.nan, inplace=True)
            neg_dec.replace(0, np.nan, inplace=True)
            neg_inc.replace(0, np.nan, inplace=True)

            sqz_inc = squeeze * increasing(squeeze)
            sqz_dec = squeeze * decreasing(squeeze)
            sqz_inc.replace(0, np.nan, inplace=True)
            sqz_dec.replace(0, np.nan, inplace=True)

            # Handle fills
            if "fillna" in kwargs:
                sqz_inc.fillna(kwargs["fillna"], inplace=True)
                sqz_dec.fillna(kwargs["fillna"], inplace=True)
                pos_inc.fillna(kwargs["fillna"], inplace=True)
                pos_dec.fillna(kwargs["fillna"], inplace=True)
                neg_dec.fillna(kwargs["fillna"], inplace=True)
                neg_inc.fillna(kwargs["fillna"], inplace=True)
            if "fill_method" in kwargs:
                sqz_inc.fillna(method=kwargs["fill_method"], inplace=True)
                sqz_dec.fillna(method=kwargs["fill_method"], inplace=True)
                pos_inc.fillna(method=kwargs["fill_method"], inplace=True)
                pos_dec.fillna(method=kwargs["fill_method"], inplace=True)
                neg_dec.fillna(method=kwargs["fill_method"], inplace=True)
                neg_inc.fillna(method=kwargs["fill_method"], inplace=True)

            df[f"SQZPRO_INC"] = sqz_inc
            df[f"SQZPRO_DEC"] = sqz_dec
            df[f"SQZPRO_PINC"] = pos_inc
            df[f"SQZPRO_PDEC"] = pos_dec
            df[f"SQZPRO_NDEC"] = neg_dec
            df[f"SQZPRO_NINC"] = neg_inc

        logger.info("Successfully calculated squeeze_pro indicator")
        return df
    
    except Exception as e:
        logger.error(f"Error calculating squeeze_pro indicator: {e}")
        return None
