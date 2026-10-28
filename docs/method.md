# Method

## Question

How do indirect (connecting) flight prices evolve relative to direct flight prices over the booking horizon, and how is that relationship shaped by quality differences (travel time) and availability constraints (direct flights selling out)?

## Hypothesis

Direct flights are the quality benchmark: passengers prefer them, so they carry a premium. As direct flights fill up and disappear, indirect itineraries should absorb displaced demand and reprice upward.

## Model

```
log(price_indirect) = α0
                    + α1 · log(DirectPrice_avg)
                    + α2 · log(DirectPrice_avg) × DaysBeforeDeparture
                    + α3 · log(TravelTime)
                    + α4 · [log(TravelTime)]²
                    + α5 · SoldOutDummy
                    + RouteFixedEffect
                    + DepartureDateFixedEffect
                    + YearFixedEffect
                    + ε
```

### Coefficient Interpretation

| Coefficient | Variable | Expected Sign | Meaning |
|---|---|---|---|
| α1 | log(DirectPrice_avg) | + | Substitution: indirect tracks direct prices |
| α2 | log(DirectPrice_avg) × DaysBeforeDeparture | ≠ 0 | Does linkage strengthen near departure? |
| α3 | log(TravelTime) | ? | Base effect of travel time on price |
| α4 | [log(TravelTime)]² | ? | Non-linear quality/cost curve |
| α5 | SoldOutDummy | + | Scarcity effect: indirect prices rise as directs sell out |

---

## Data Pipeline Plan

### Step 1 — Clean Raw Data
- Rename columns to consistent names (e.g. `daysbeforedeparture` → `DaysBeforeDeparture`)
- Strip currency symbols from price (2026 data has "€" prefix)
- Parse travel time from "Xu Ym" format to minutes

### Step 2 — Remove Duplicates
- Remove true duplicates (all columns identical)
- Group by: FlightCode, Route, DepartureDate, DaysBeforeDeparture, TravelTime
- Keep minimum price per group

### Step 3 — Split Direct / Indirect
- Direct: `ConnectionNumber == 0`
- Indirect: `ConnectionNumber > 0`

### Step 4 — Construct DirectPrice Benchmark
- For each (Route, DepartureDate, DaysBeforeDeparture):
  - Compute `DirectPrice_avg` = mean price of direct flights

### Step 5 — Create SoldOutDummy
- For each (Route, DepartureDate):
  - Establish initial direct flight count (at max DaysBeforeDeparture)
  - For each day: `SoldOutDummy = 1` if current count < initial count

### Step 6 — Merge
- Join `DirectPrice_avg` and `SoldOutDummy` into the indirect flights dataset

### Step 7 — Final Dataset Variables
| Variable | Description |
|---|---|
| FlightCode | Traceability |
| price | Indirect flight price |
| DirectPrice_avg | Avg direct price on same route/date/day |
| DaysBeforeDeparture | Days until departure at scrape time |
| TravelTime | In minutes (parsed from raw) |
| log(TravelTime) | Log transformation |
| log(TravelTime)² | Non-linear term |
| SoldOutDummy | 1 if any direct flights have disappeared |
| Route | Route identifier |
| DepartureDate | Date of departure |
| Year | 2023 or 2026 |

### Step 8 — Create Log Variables
- `log(price)`, `log(DirectPrice_avg)`, `log(TravelTime)`, `[log(TravelTime)]²`
- Interaction: `log(DirectPrice_avg) × DaysBeforeDeparture`
- Only apply to positive values; investigate zeros/negatives before dropping

---

## Modelling Plan

1. **OLS** with route, departure date, and year fixed effects
2. **Cluster standard errors** at the route level
3. Interpret: signs, significance, magnitudes of α1–α5
4. Estimate the same specification on 2023, 2026 and pooled (year FE)

---
