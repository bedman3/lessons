# ML Research Memo

Use one memo per experiment family. Keep the original holdout sealed until the choices below are frozen. A completed memo records what was tried and what the evidence supports; it does not turn a model score into a profit claim.

## Question and decision

- Research question:
- Potential decision this could inform:
- What evidence would change the next step?
- What this experiment cannot establish:

## Prediction contract

- Unit of observation (asset-date, trade, etc.):
- Decision timestamp and timezone:
- Feature availability cutoff:
- Target definition, units, and start/end interval:
- Universe and inclusion/exclusion rules:
- Label availability date for each observation:
- Unresolved assumptions and checks:

## Data and provenance

- Data source, snapshot/version, and retrieval date:
- Event time versus availability time handling:
- Revision, missingness, duplicate, and survivorship checks:
- Code revision and environment/package versions:

## Predeclared evaluation

- Development dates:
- Validation dates and rolling/expanding scheme:
- Frozen original holdout dates:
- Purging rule based on label end; embargo rationale if used:
- Train-only learned preprocessing:
- Primary metric and aggregation unit (for example, mean per-date MAE):
- Diagnostic metrics and subgroup/date slices:
- Baselines and why each matches the decision:
- Cost assumptions and stress cases, if relevant:
- Uncertainty method, block unit/length, and limitations:

## Trial log

| Trial | Target/features/model/configuration | Dates | Primary metric | Date-block results | Decision / reason |
|---|---|---|---|---|---|
| 1 | | | | | |

Record meaningful attempts, including unsuccessful variants and any change prompted by validation results. Do not use the final holdout to select among variants.

## Results and interpretation

- Data and split checks completed:
- Baseline result:
- Candidate result and matched difference:
- Number of dates and observations:
- Date/regime breakdown and uncertainty:
- Sensitivity to plausible costs, delay, and universe changes:
- Known failure cases and limitations:
- Did anyone inspect the original holdout? If yes, how did that affect decisions?

## Decision

- Supported conclusion, stated narrowly:
- Unsupported conclusions to avoid (including profitability or qualification claims):
- Next experiment, if justified:
- Reproduction instructions and artefact locations:
