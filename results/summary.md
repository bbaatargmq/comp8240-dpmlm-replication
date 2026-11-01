# DP-MLM project update - results summary

Device: cuda | QUICK=True | eps=[10, 25, 50, 100, 250]

## Utility on original + existing data

| dataset   | epsilon   |    BLEU |   CosSim |   Acc |   %words changed |
|:----------|:----------|--------:|---------:|------:|-----------------:|
| SST-2     | original  | 100     |    1     | 0.913 |            0     |
| SST-2     | 10        |   6.808 |    0.176 | 0.52  |           66.928 |
| SST-2     | 25        |  10.098 |    0.366 | 0.64  |           67.227 |
| SST-2     | 50        |  15.902 |    0.551 | 0.78  |           65.601 |
| SST-2     | 100       |  19.266 |    0.602 | 0.833 |           65.16  |
| SST-2     | 250       |  18.839 |    0.614 | 0.773 |           64.517 |
| IMDb      | original  | 100     |    1     | 0.753 |            0     |
| IMDb      | 10        |  12.794 |    0.178 | 0.547 |           85.302 |
| IMDb      | 25        |  15.594 |    0.409 | 0.647 |           84.036 |
| IMDb      | 50        |  22.008 |    0.616 | 0.693 |           81.712 |
| IMDb      | 100       |  24.825 |    0.679 | 0.74  |           81.246 |
| IMDb      | 250       |  25.33  |    0.691 | 0.713 |           80.282 |


## Constructed Reddit dataset - construction funnel

|                         |   posts |
|:------------------------|--------:|
| posts parsed            |  129722 |
| poster tag in title     |   38366 |
| no explicit age in body |   36766 |
| 15-60 words             |   36525 |
| deduplicated            |   36102 |
| balanced sample         |    4000 |


Title-vs-body label agreement: 0.988 (n=994)

## Attack results (macro-F1 %)

|                                      |   original |    10 |    25 |    50 |   100 |   250 |
|:-------------------------------------|-----------:|------:|------:|------:|------:|------:|
| ('utility: subreddit', 'static')     |      48.05 | 54.98 | 48.32 | 48.19 | 54.98 | 47.92 |
| ('utility: subreddit', 'adaptive')   |      48.32 | 54.98 | 48.19 | 48.32 | 48.32 | 48.32 |
| ('privacy: gender', 'static')        |      83.5  | 78    | 77.96 | 83.99 | 77.49 | 79.93 |
| ('privacy: gender', 'adaptive')      |      81.42 | 78.5  | 74.49 | 75.5  | 78.47 | 78.98 |
| ('privacy: age bracket', 'static')   |      66.19 | 55.72 | 57.73 | 60.14 | 58.59 | 59.47 |
| ('privacy: age bracket', 'adaptive') |      63.77 | 58.59 | 56.18 | 56.93 | 59.67 | 58.45 |


## Relative Gain

|                             |    10 |    25 |     50 |   100 |   250 |
|:----------------------------|------:|------:|-------:|------:|------:|
| ('static', 'gender')        | 0.21  | 0.072 | -0.003 | 0.216 | 0.04  |
| ('static', 'age bracket')   | 0.302 | 0.133 |  0.094 | 0.259 | 0.099 |
| ('adaptive', 'gender')      | 0.174 | 0.082 |  0.073 | 0.036 | 0.03  |
| ('adaptive', 'age bracket') | 0.219 | 0.116 |  0.107 | 0.064 | 0.083 |