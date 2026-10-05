"""0050 成分股調整的時間表。

* 公告日：3/6/9/12 月的第一個星期五；若非交易日則順延到下一個交易日。
* 決策點：公告日收盤（此時新成分股名單已公開）。
* 持有期：公告日收盤 → 下一季公告日收盤（相鄰季度首尾相接，無空窗）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


def first_friday(year: int, month: int) -> datetime:
    first = datetime(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7)


def announcement_date(quarter: str, trading_dates: pd.DatetimeIndex) -> pd.Timestamp:
    year, month = map(int, quarter.split("_"))
    d = pd.Timestamp(first_friday(year, month))
    pos = trading_dates.searchsorted(d)
    return trading_dates[pos]


@dataclass(frozen=True)
class QuarterWindow:
    quarter: str
    announcement: pd.Timestamp      # 決策日（公告日收盤）
    hold_end: pd.Timestamp          # 下一季公告日（持有到該日收盤）

    def as_dict(self) -> dict:
        return {"quarter": self.quarter,
                "announcement": self.announcement.date().isoformat(),
                "hold_end": self.hold_end.date().isoformat()}


def build_schedule(quarters: list[str], next_after_last: str,
                   trading_dates: pd.DatetimeIndex) -> list[QuarterWindow]:
    ann = [announcement_date(q, trading_dates) for q in quarters + [next_after_last]]
    return [QuarterWindow(q, ann[i], ann[i + 1]) for i, q in enumerate(quarters)]
