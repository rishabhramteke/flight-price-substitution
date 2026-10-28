# Data Analysis

## Overview

Two datasets covering the same European short-haul routes in different years:

| | AMD.csv (2023) | flights_output_20260409_125933.csv (2026) |
|---|---|---|
| Rows | 445,385 | 11,947 (scraping in progress) |
| Columns | 11 | 21 |
| Scrape dates | 16 Mar – 21 Apr 2023 | 9 Apr 2026 (1 day so far) |
| Departure dates | 15 Apr – 21 Apr 2023 | 15 Apr – 21 Apr 2026 |
| Booking horizon | 1–36 days before departure | 5–11 days before departure (growing) |
| Routes | 155 | 57 |
| Direct flights | 90,744 (20.4%) | 1,511 (12.6%) |
| Indirect flights | 354,641 (79.6%) | 10,436 (87.4%) |
| Missing values | None | Several columns entirely empty (by design) |

---

## 2023 Data — AMD.csv

### Columns

| Column | Type | Notes |
|---|---|---|
| flightcode | int | Numeric flight identifier |
| airline | string | 77 unique airlines |
| lccdum | int (0/1) | Low-cost carrier dummy |
| route | string | e.g. "AMS-CPH" |
| lccpercentage | float | Share of LCCs on the route |
| departuredate | string | Format: DD/MM/YYYY |
| connectionnumber | int | 0 = direct, 1/2/3 = indirect |
| traveltime | string | Format: "Xu Ym" (e.g. "1u 30m") |
| scrapedate | string | Format: DD/MM/YYYY |
| price | int | Price in EUR (no symbol) |
| daysbeforedeparture | int | Days between scrape and departure |

### Data Quality
- **No missing values** in any column — clean dataset ready for processing
- `traveltime` needs parsing to numeric (minutes)
- `departuredate` and `scrapedate` need datetime conversion

### Routes
- 155 routes total; busiest: IST-TLS (6,419 rows), TLS-IST (6,070), AMS-BGO (3,554)
- Routes appear roughly symmetric (e.g. AMS-CPH and CPH-AMS both present)

### Connection Numbers
| Connections | Count | % |
|---|---|---|
| 0 (direct) | 90,744 | 20.4% |
| 1 (one stop) | 303,201 | 68.1% |
| 2 (two stops) | 51,206 | 11.5% |
| 3 (three stops) | 234 | 0.05% |

### Price Distribution
| Stat | Value (EUR) |
|---|---|
| Min | 1 |
| 25th pct | 153 |
| Median | 229 |
| Mean | 257 |
| 75th pct | 328 |
| Max | 999 |

> Note: €1 minimum is suspicious — may be error or placeholder. Investigate before log-transforming.

### Booking Horizon
- Range: 1–36 days before departure
- Mean: ~17 days, roughly uniform distribution

### Top Airlines (2023)
1. Lufthansa — 90,539
2. Scandinavian Airlines — 43,526
3. KLM — 41,722
4. Air France — 31,683
5. SWISS — 23,444

---

## 2026 Data — flights_output_20260409_125933.csv

### Columns

| Column | Type | Status | Notes |
|---|---|---|---|
| FlightCode | string | Partial (61 missing) | e.g. "EI 657" (letter+number format) |
| airline | string | Complete | |
| LCCdum | float | **All empty** | To be computed post-scrape |
| Route | string | Complete | e.g. "FRA-DUB" |
| Origin | string | Complete | New vs 2023 |
| Destination | string | Complete | New vs 2023 |
| LCCpercentage | float | **All empty** | To be computed post-scrape |
| HHI | float | **All empty** | Market concentration index — new variable |
| NumberOfCarrier | float | Almost all empty (11 non-null) | To be computed |
| DepartureDate | string | Complete | Format: M/D/YYYY |
| DepartureTime | string | Complete | Format: HH:MM — new vs 2023 |
| Arrival Time | string | Complete | New vs 2023 |
| ConnectionNumber | int | Complete | 0/1/2/4 |
| TravelTime | string | Complete | Same "Xu Ym" format as 2023 |
| ScrapeDate | string | Complete | Format: M/D/YYYY |
| price | string | Complete | Has "€" prefix — needs stripping |
| Date before departure | int | Complete | Renamed `daysbeforedeparture` equivalent |
| FareMean | float | **All empty** | To be computed post-scrape |
| SoldOutDate | float | **All empty** | To be computed post-scrape |
| PercentDeviationFromFareMean | float | **All empty** | To be computed post-scrape |
| DeviationFromSoldOutDate | float | **All empty** | To be computed post-scrape |

### Empty Columns (by design — computed after full scrape)
These 7 columns will be filled once scraping is complete:
- `LCCdum`, `LCCpercentage`, `HHI`, `FareMean`, `SoldOutDate`, `PercentDeviationFromFareMean`, `DeviationFromSoldOutDate`

### Scraping Status
- Only **1 scrape date** so far: 9 Apr 2026
- Booking horizon: 5–11 days before departure (scraping started ~11 days out)
- Full window expected: up to 36 days before departure
- Routes: 57 (subset of 2023's 155)

### Connection Numbers (2026)
| Connections | Count | % |
|---|---|---|
| 0 (direct) | 1,511 | 12.6% |
| 1 (one stop) | 9,498 | 79.5% |
| 2 (two stops) | 926 | 7.8% |
| 4 (four stops) | 12 | 0.1% |

> ConnectionNumber = 4 is unusual — check if these are data errors or genuine itineraries.

### Price
- Stored as string with "€" prefix (e.g. "€139")
- **Needs stripping** before any numeric operations

### Booking Horizon (so far)
- Range: 5–11 days (will grow as scraping continues)
- Mean: ~8 days

---

## Key Differences Between 2023 and 2026 Data

| Aspect | 2023 | 2026 |
|---|---|---|
| Price format | Integer (EUR, no symbol) | String with "€" prefix |
| FlightCode format | Numeric integer | Alphanumeric string ("EI 657") |
| Column names | lowercase, no spaces | Mixed case, some with spaces |
| DaysBeforeDeparture name | `daysbeforedeparture` | `Date before departure` |
| Departure/Arrival times | Not available | Available |
| Market structure (HHI) | Not available | Planned (currently empty) |
| Data completeness | Fully clean | Partially populated (scraping in progress) |
| Routes covered | 155 | 57 (subset) |
| LCC dummy | Available | Not yet populated |

---

## Pre-Processing Flags / Warnings

1. **Price cleaning (2026)**: Strip "€" from `price` column before any numeric operations.
2. **Suspicious price floor (2023)**: Minimum price = €1 — investigate whether these are errors.
3. **ConnectionNumber = 4 (2026)**: 12 rows with 4 connections — verify if valid or data artifact.
4. **FlightCode 61 missing (2026)**: Small number; can flag or impute from airline + time.
5. **Column name standardisation**: Before merging years, align column names (e.g. `daysbeforedeparture` ↔ `Date before departure`).
6. **Date format difference**: 2023 uses DD/MM/YYYY, 2026 uses M/D/YYYY — parse carefully to avoid month/day swap errors.
7. **TravelTime parsing**: "Xu Ym" format in both datasets; convert to total minutes for log transformation.
8. **Log transformation safety**: Ensure price > 0 and TravelTime > 0 before applying log. With price = €1 minimum in 2023, log(1) = 0 which is valid but may warrant investigation.

---

## Research Design Questions & Limitations
