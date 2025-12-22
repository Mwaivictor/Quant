"""
Time Alignment Module

Converts all timestamps to UTC and aligns bars to canonical time grids.
This is the first critical step in the cleaning pipeline.

Rules:
    - All output timestamps must be UTC
    - Bars must align to fixed schedules (no partial times)
    - Missing bars are explicitly inserted with NULL OHLC
    - No forward-filling or interpolation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import List, Optional, Tuple
import logging

from arbitrex.clean_data.config import CleanDataConfig, TimeAlignment

LOG = logging.getLogger(__name__)


class TimeAligner:
    """
    Aligns raw bars to canonical UTC time grids.
    
    Philosophy:
        - Trust nothing about input timestamps
        - Enforce strict schedule alignment
        - Make missing data explicit
    """
    
    def __init__(self, config: CleanDataConfig):
        self.config = config
        self.time_alignment = config.time_alignment
    
    def align_to_grid(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Align bars to canonical time grid.
        
        Args:
            df: Raw bars with timestamp column
            symbol: Symbol being processed
            timeframe: Timeframe (1H, 4H, 1D)
        
        Returns:
            DataFrame with canonical timestamps and missing bars inserted
        
        Process:
            1. Convert timestamps to UTC if not already
            2. Determine canonical schedule for timeframe
            3. Generate expected timestamps
            4. Left join expected with actual
            5. Mark missing bars
        """
        if df.empty:
            LOG.warning(f"Empty dataframe for {symbol} {timeframe}")
            return df
        
        # Step 1: Ensure timestamps are UTC
        df = self._ensure_utc(df)
        
        # Step 2: Get canonical schedule
        schedule = self._get_schedule(timeframe)
        if schedule is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        # Step 3: Generate expected timestamps
        expected_timestamps = self._generate_expected_timestamps(
            start_date=df["timestamp_utc"].min(),
            end_date=df["timestamp_utc"].max(),
            schedule=schedule,
            timeframe=timeframe
        )
        
        # Step 4: Create expected grid dataframe
        expected_df = pd.DataFrame({
            "timestamp_utc": expected_timestamps,
            "symbol": symbol,
            "timeframe": timeframe,
        })
        
        # Step 5: Left join to identify missing bars
        aligned_df = expected_df.merge(
            df,
            on=["timestamp_utc", "symbol", "timeframe"],
            how="left"
        )
        
        # Step 6: Mark missing bars
        aligned_df["is_missing"] = aligned_df["close"].isna()
        
        LOG.info(
            f"Aligned {symbol} {timeframe}: "
            f"{len(df)} raw bars → {len(aligned_df)} expected bars "
            f"({aligned_df['is_missing'].sum()} missing)"
        )
        
        return aligned_df
    
    def _ensure_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure timestamp column is in UTC.
        
        Handles:
            - timestamp_utc already present (preferred)
            - timestamp in broker time (convert using offset)
            - No timezone info (assume UTC with warning)
        """
        # Check if timestamp_utc already exists
        if "timestamp_utc" in df.columns:
            # Ensure it's datetime64[ns, UTC]
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            return df
        
        # Check for raw timestamp
        if "timestamp" in df.columns:
            # Convert to datetime
            ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            
            # If broker_utc_offset_hours available, adjust
            if "broker_utc_offset_hours" in df.columns:
                offset_hours = df["broker_utc_offset_hours"].iloc[0]
                ts = ts - pd.Timedelta(hours=offset_hours)
                LOG.debug(f"Converted broker time to UTC (offset: {offset_hours}h)")
            else:
                LOG.warning("No broker offset found, assuming timestamp is UTC")
            
            df["timestamp_utc"] = ts
            return df
        
        raise ValueError("No timestamp column found in dataframe")
    
    def _get_schedule(self, timeframe: str) -> Optional[List[time]]:
        """Get canonical schedule for timeframe"""
        if timeframe == "1H":
            return self.time_alignment.schedule_1H
        elif timeframe == "4H":
            return self.time_alignment.schedule_4H
        elif timeframe == "1D":
            return self.time_alignment.schedule_1D
        else:
            return None
    
    def _generate_expected_timestamps(
        self,
        start_date: datetime,
        end_date: datetime,
        schedule: List[time],
        timeframe: str
    ) -> List[datetime]:
        """
        Generate all expected timestamps between start and end dates.
        
        Args:
            start_date: First bar date
            end_date: Last bar date
            schedule: List of times for this timeframe
            timeframe: Timeframe identifier
        
        Returns:
            List of expected UTC timestamps
        """
        timestamps = []
        
        # Normalize to date boundaries
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            for scheduled_time in schedule:
                # Combine date with scheduled time
                dt = datetime.combine(current_date, scheduled_time)
                
                # Make timezone-aware (UTC)
                dt = dt.replace(tzinfo=pd.Timestamp.now(tz="UTC").tzinfo)
                
                # Only include if within range
                if start_date <= dt <= end_date:
                    timestamps.append(dt)
            
            # Increment by one day
            current_date += timedelta(days=1)
        
        return sorted(timestamps)
    
    def validate_alignment(self, df: pd.DataFrame, timeframe: str) -> Tuple[bool, List[str]]:
        """
        Validate that all timestamps conform to canonical schedule.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if df.empty:
            return True, issues
        
        # Get schedule
        schedule = self._get_schedule(timeframe)
        if schedule is None:
            issues.append(f"Unknown timeframe: {timeframe}")
            return False, issues
        
        # Check all timestamps match schedule times
        for ts in df["timestamp_utc"]:
            ts_time = ts.time()
            if ts_time not in schedule:
                issues.append(
                    f"Timestamp {ts} does not match schedule for {timeframe}"
                )
        
        # Check timestamps are sorted
        if not df["timestamp_utc"].is_monotonic_increasing:
            issues.append("Timestamps are not sorted")
        
        # Check no duplicates
        if df["timestamp_utc"].duplicated().any():
            issues.append("Duplicate timestamps found")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            LOG.warning(f"Alignment validation failed: {len(issues)} issues found")
        
        return is_valid, issues
    
    def get_alignment_report(self, df: pd.DataFrame) -> dict:
        """
        Generate alignment statistics report.
        
        Returns:
            Dictionary with alignment metrics
        """
        if df.empty:
            return {
                "total_bars": 0,
                "missing_bars": 0,
                "missing_percentage": 0.0,
                "timestamp_range": None,
            }
        
        return {
            "total_bars": len(df),
            "missing_bars": df["is_missing"].sum(),
            "missing_percentage": df["is_missing"].mean() * 100,
            "timestamp_range": (
                df["timestamp_utc"].min().isoformat(),
                df["timestamp_utc"].max().isoformat()
            ),
            "first_bar_missing": df["is_missing"].iloc[0] if len(df) > 0 else None,
            "last_bar_missing": df["is_missing"].iloc[-1] if len(df) > 0 else None,
        }
