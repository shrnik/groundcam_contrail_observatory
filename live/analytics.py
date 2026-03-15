import csv
import glob
import logging
import os
from datetime import date, datetime

logger = logging.getLogger(__name__)

_COLUMNS = ["timestamp", "camera_name", "ident", "px", "py", "alt_m", "contrail", "image_url"]


def log(
    results: list[tuple],
    output_dir: str,
    camera_name: str = "",
    image_url: str = "",
) -> None:
    """Append result rows to today's daily CSV file."""
    if not results:
        return
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().isoformat()
    path = os.path.join(output_dir, f"{today}.csv")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if f.tell() == 0:
            writer.writeheader()
        for timestamp, ident, px, py, alt_m, contrail in results:
            writer.writerow({
                "timestamp": timestamp.isoformat(),
                "camera_name": camera_name,
                "ident": ident,
                "px": f"{px:.1f}",
                "py": f"{py:.1f}",
                "alt_m": f"{alt_m:.0f}",
                "contrail": int(contrail),
                "image_url": image_url,
            })


def daily_summary(target_date: date, output_dir: str) -> dict:
    """Count unique transponders with contrail=True vs total for a given date."""
    path = os.path.join(output_dir, f"{target_date.isoformat()}.csv")
    if not os.path.exists(path):
        return {"date": target_date.isoformat(), "total": 0, "contrail": 0}

    total, contrail = set(), set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total.add(row["ident"])
            if row["contrail"] == "1":
                contrail.add(row["ident"])

    return {
        "date": target_date.isoformat(),
        "total_aircraft": len(total),
        "contrail_aircraft": len(contrail),
    }


def monthly_summary(year: int, month: int, output_dir: str) -> list[dict]:
    """Aggregate daily summaries for every CSV in a given year/month."""
    pattern = os.path.join(output_dir, f"{year}-{month:02d}-*.csv")
    summaries = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        d = date.fromisoformat(fname.removesuffix(".csv"))
        summaries.append(daily_summary(d, output_dir))
    return summaries
