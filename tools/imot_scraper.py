from __future__ import annotations

import logging
import pathlib
import re
import time

import pandas as pd
import requests

ENDPOINT = "https://www.imot.bg/sredni-ceni"
USER_AGENT = (
    "Mozilla/5.0 (sofia-housing-course-project)"
)
REQUEST_DELAY_S = 2.0
PAGE_ENCODING = "windows-1251"

TRANSACTION_TYPE_CODES = {"sale": "0", "rent": "1", "lots": "2"}

SCRAPER_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SCRAPER_DIR / "cache"
DEFAULT_CSV_PATH = SCRAPER_DIR / "data" / "imot_avg_prices.csv"

logger = logging.getLogger(__name__)


def _cache_key(tx_type: str, town: str, year: int, date: str | None) -> pathlib.Path:
    safe_town = re.sub(r"[^A-Za-z0-9_-]", "_", _translit(town))
    if date is None:
        return DEFAULT_CACHE_DIR / f"sredni_{tx_type}_{safe_town}_{year}_index.html"
    iso = _date_to_iso(date)
    return DEFAULT_CACHE_DIR / f"sredni_{tx_type}_{safe_town}_{iso}.html"


def _translit(s: str) -> str:
    table = str.maketrans(
        {
            "А": "A",
            "Б": "B",
            "В": "V",
            "Г": "G",
            "Д": "D",
            "Е": "E",
            "Ж": "Zh",
            "З": "Z",
            "И": "I",
            "Й": "Y",
            "К": "K",
            "Л": "L",
            "М": "M",
            "Н": "N",
            "О": "O",
            "П": "P",
            "Р": "R",
            "С": "S",
            "Т": "T",
            "У": "U",
            "Ф": "F",
            "Х": "H",
            "Ц": "Ts",
            "Ч": "Ch",
            "Ш": "Sh",
            "Щ": "Sht",
            "Ъ": "A",
            "Ь": "Y",
            "Ю": "Yu",
            "Я": "Ya",
        }
    )
    return s.translate(table)


def _date_to_iso(date_dmy: str) -> str:
    d, m, y = date_dmy.split(".")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def fetch_snapshot(
    tx_type_code: str,
    town: str,
    year: int,
    date: str | None,
    cache_dir: pathlib.Path,
    force_refresh: bool,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_key(tx_type_code, town, year, date)

    if cache_file.exists() and not force_refresh:
        return cache_file.read_text(encoding="utf-8")

    data = {"pn": tx_type_code, "town": town, "year": str(year)}
    if date is not None:
        data["date"] = date

    logger.info("POST year=%d date=%s", year, date or "<index>")
    resp = requests.post(
        ENDPOINT,
        data=data,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "bg,en;q=0.8"},
        timeout=30,
    )
    resp.raise_for_status()
    resp.encoding = PAGE_ENCODING
    text = resp.text
    cache_file.write_text(text, encoding="utf-8")
    time.sleep(REQUEST_DELAY_S)
    return text


_DATE_OPTION_RE = re.compile(
    r'<option(?:\s+selected)?\s+value="(\d{1,2}\.\d{1,2}\.\d{4})"'
)
_TABLE_RE = re.compile(
    r'<table class="sredni-ceni-\d+"[^>]*>(.*?)</table>',
    re.DOTALL,
)
_ROW_CELLS_RE = re.compile(
    r"<tr>\s*" + r"\s*".join([r"<td>(.*?)</td>"] * 8) + r"\s*</tr>",
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_DISTRICT_TEXT_RE = re.compile(r">([^<]+)</a>")


def parse_dates_for_year(html: str, year: int) -> list[str]:
    dates = []
    for m in _DATE_OPTION_RE.finditer(html):
        d = m.group(1)
        if d.endswith(f".{year}"):
            dates.append(d)
    seen, out = set(), []
    for d in dates:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _strip_tags(s: str) -> str:
    return _TAG_RE.sub("", s).replace("\xa0", " ").strip()


def _clean_number(cell_html: str) -> float | None:
    text = _strip_tags(cell_html)
    text = text.replace(" ", "").replace("\xa0", "")
    if not text or text == "-":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _district_from_cell(cell_html: str) -> str:
    m = _DISTRICT_TEXT_RE.search(cell_html)
    if m:
        return m.group(1).strip()
    return _strip_tags(cell_html)


def parse_snapshot(html: str) -> list[dict]:
    table_m = _TABLE_RE.search(html)
    if not table_m:
        return []
    table_html = table_m.group(1)

    records = []
    for m in _ROW_CELLS_RE.finditer(table_html):
        cells = m.groups()
        district = _district_from_cell(cells[0])
        if not district:
            continue
        records.append(
            {
                "district": district,
                "price_1room_eur": _clean_number(cells[1]),
                "eur_per_sqm_1room": _clean_number(cells[2]),
                "price_2room_eur": _clean_number(cells[3]),
                "eur_per_sqm_2room": _clean_number(cells[4]),
                "price_3room_eur": _clean_number(cells[5]),
                "eur_per_sqm_3room": _clean_number(cells[6]),
                "eur_per_sqm_overall": _clean_number(cells[7]),
            }
        )
    return records


def scrape(
    start_year: int = 2015,
    end_year: int | None = None,
    transaction_type: str = "sale",
    town: str = "София",
    cache_dir: pathlib.Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if transaction_type not in TRANSACTION_TYPE_CODES:
        raise ValueError(
            f"transaction_type must be one of {list(TRANSACTION_TYPE_CODES)}; "
            f"got {transaction_type!r}"
        )
    tx_code = TRANSACTION_TYPE_CODES[transaction_type]
    if end_year is None:
        end_year = pd.Timestamp.now().year
    if start_year > end_year:
        raise ValueError(f"start_year {start_year} > end_year {end_year}")

    all_rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        index_html = fetch_snapshot(
            tx_code,
            town,
            year,
            date=None,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        dates = parse_dates_for_year(index_html, year)
        if not dates:
            logger.warning("Year %d: no dates in dropdown — skipping.", year)
            continue
        logger.info("Year %d: %d weekly snapshots available.", year, len(dates))

        for date in dates:
            html = fetch_snapshot(
                tx_code,
                town,
                year,
                date=date,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
            rows = parse_snapshot(html)
            if not rows:
                logger.warning("Year %d %s: parsed 0 rows.", year, date)
                continue
            iso = _date_to_iso(date)
            for r in rows:
                r["snapshot_date"] = iso
                r["year"] = year
                r["transaction_type"] = transaction_type
                r["town"] = town
            all_rows.extend(rows)

    if not all_rows:
        logger.error("Collected zero rows. Check connectivity / cache.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df[
        [
            "snapshot_date",
            "year",
            "transaction_type",
            "town",
            "district",
            "price_1room_eur",
            "eur_per_sqm_1room",
            "price_2room_eur",
            "eur_per_sqm_2room",
            "price_3room_eur",
            "eur_per_sqm_3room",
            "eur_per_sqm_overall",
        ]
    ]
    df = df.sort_values(["snapshot_date", "district"]).reset_index(drop=True)
    return df


def save_dataset(
    df: pd.DataFrame, out_path: pathlib.Path = DEFAULT_CSV_PATH
) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def _build_cli():
    import argparse

    p = argparse.ArgumentParser(
        description="Polite, cached scraper for imot.bg /sredni-ceni "
        "(weekly aggregated Sofia district prices). "
        "Writes data/imot_avg_prices.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Inclusive. Defaults to the current year.",
    )
    p.add_argument(
        "--transaction-type", choices=list(TRANSACTION_TYPE_CODES), default="sale"
    )
    p.add_argument(
        "--town",
        default="София",
        help="Bulgarian city name in Cyrillic (e.g. София, Пловдив).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the HTML cache and re-fetch every page.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args = _build_cli().parse_args()
    df = scrape(
        start_year=args.start_year,
        end_year=args.end_year,
        transaction_type=args.transaction_type,
        town=args.town,
        force_refresh=args.no_cache,
    )
    if df.empty:
        raise SystemExit(1)
    out = save_dataset(df)
    print(f"\nSaved {len(df):,} rows to {out}")
    print(f"Date range : {df['snapshot_date'].min()} .. {df['snapshot_date'].max()}")
    print(f"Districts  : {df['district'].nunique()}")
    print(f"Snapshots  : {df['snapshot_date'].nunique()}")
    print(f"\nSample:\n{df.head(3).to_string(index=False)}")
