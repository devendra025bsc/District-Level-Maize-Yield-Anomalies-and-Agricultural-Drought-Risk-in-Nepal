# Recovering verified district-year maize yield (Critical Problem C1-C6)

This is the single most important data gap in the manuscript. No public,
programmatically-queryable API exists for Nepal's official district-level
agricultural statistics, so this step is **manual** and cannot be scripted
end-to-end. Do not substitute a reconstructed/estimated series and label it
as verified (Integrity Class A) -- if you cannot complete this step, keep the
yield field as Class D (reconstructed) and say so explicitly in the
manuscript's Limitations section, exactly as the current draft already does.

## What to look up

For each of the six districts (Jhapa, Ilam, Bhojpur, Morang, Dhankuta,
Sunsari) and each of the five event years (2015, 2016, 2018, 2022, 2024),
find:

- Maize area harvested (ha)
- Maize production (metric tons)
- Maize yield (t/ha) -- usually production / area, sometimes reported directly

## Where to look

1. **MoALD "Statistical Information on Nepalese Agriculture"** (annual
   series). Published by the Ministry of Agriculture and Livestock
   Development, Planning and Development Cooperation Coordination Division.
   - Portal: https://moald.gov.np/publication/ (browse for the relevant
     fiscal-year edition; Nepali fiscal year 2071/72 ~ 2015, 2072/73 ~ 2016,
     2074/75 ~ 2018, 2078/79 ~ 2022, 2080/81 ~ 2024).
   - Also check https://nsa.moald.gov.np/ if it is live.
   - District-level tables are usually in an appendix titled something like
     "Area, Production and Yield of Major Crops by District."

2. **Central Bureau of Statistics (CBS) Nepal** -- National Sample Census of
   Agriculture, or Statistical Yearbook of Nepal.
   - https://cbs.gov.np/
   - Use as a cross-check against MoALD figures; if the two disagree, report
     both and flag the discrepancy (do not silently average them).

3. **Provincial Ministry of Land Management, Agriculture and Cooperatives,
   Province 1 (Koshi)** -- may publish province-specific district breakdowns
   not in the national MoALD compilation.

4. **FAO GIEWS Country Brief: Nepal** -- national-level only, useful for
   sanity-checking district totals against the province/national aggregate,
   not a substitute for district-level figures.

## How to record what you find

Fill in `yield_moald_TEMPLATE.csv` in this folder, one row per
district-year. Leave `source_page` and `source_edition` filled in precisely
enough that a reviewer could find the same number -- this is what makes the
value Integrity Class A (verified observed data) rather than Class D.

If a specific district-year cell cannot be found in any of the sources
above, leave `yield_t_ha` blank and set `status` to `NOT_RECOVERED` rather
than filling in an estimate. The downstream analysis scripts are written to
handle missing cells honestly (smaller n, not an imputed value) unless you
explicitly choose and document an imputation method.

## If verified data cannot be recovered for some/all district-years

This is a legitimate, expected outcome, not a failure to document. In that
case:

- Keep using the MoALD-consistent *reconstructed* series already documented
  in Section 2.7 of the manuscript, explicitly labelled Integrity Class D.
- State in Data Availability / Limitations exactly which district-years are
  Class A (verified) vs Class D (reconstructed) -- a mixed table is fine and
  more honest than an all-or-nothing choice.
- Do not run the "verified yield" branch of `08_build_temporal_coverage_matrix.py`
  for district-years still at Class D.
