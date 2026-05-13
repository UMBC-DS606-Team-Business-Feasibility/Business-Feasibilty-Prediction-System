# Metropolitan Business Feasibility Prediction System

## Ranking and Predicting High-Opportunity U.S. Metropolitan Areas from County Business Patterns, ACS, and BLS data

## Executive Summary
- Official CBP raw rows available: 577,424 across 925 metro/micro areas
- Official ACS metro areas available: 393 metro areas
- CBP + ACS + BLS overlap: 386 metropolitan areas
- Final analysis sample: all 386 overlapping metros
- Raw CBP rows used before aggregation: 369496
- Positive class definition: top 30% by opportunity score

## Opportunity Score
- 25% median household income
- 25% inverse unemployment rate
- 20% annual payroll per employee
- 15% establishments per 10,000 residents
- 15% industry entropy (business diversity)

## Modeling Setup
- Unit of analysis: metropolitan area
- Classification target: high-opportunity metro
- Regression target: 2024 unemployment rate
- Features: CBP structure metrics and 2-digit industry employment shares
- Split: 60/20/20 train/validation/test

## Business-Specific Feasibility Layer
- Broad business types supported: Restaurant / Food Service, Retail Store, Health Clinic / Care Service, Professional Services Firm, Salon / Repair / Personal Services
- Business feasibility score = 70% modeled metro market strength + 30% category fit from CBP sector presence
- Category fit is based on sector employment share and sector establishments per 10,000 residents
- Additional weekly update: recent unemployment change (2024 - 2023) added to the dataset for metro lookup and future model experiments
- City-to-metro lookup aliases generated: 856

## Best Model Results
- Best classifier: LogisticRegression (tuned)
- Test Accuracy: 0.8333
- Test F1: 0.7451
- Test ROC-AUC: 0.9020
- 5-fold CV F1: 0.7877 +/- 0.0891
- Best regressor: RandomForestRegressor
- Test RMSE: 1.4007
- Test MAE: 0.7643
- Test R2: 0.5128
- 5-fold CV R2: 0.3556 +/- 0.0738

## Top Business-Type Examples
- Restaurant / Food Service: Bozeman, MT with score=82.7
- Retail Store: Bozeman, MT with score=84.7
- Health Clinic / Care Service: Boulder, CO with score=82.4
- Professional Services Firm: Boulder, CO with score=87.7
- Salon / Repair / Personal Services: Washington-Arlington-Alexandria, DC-VA-MD-WV with score=84.0
- Technology / IT Services: San Jose-Sunnyvale-Santa Clara, CA with score=96.9
- Education / Childcare Services: Boulder, CO with score=80.2
- Construction / Trades: Bozeman, MT with score=90.1
- Finance / Insurance Services: Bloomington, IL with score=83.3

## Top 10 Metros
- San Jose-Sunnyvale-Santa Clara, CA: score=0.7229, income=164801, unemployment=4.20, est/10k=252.18
- Midland, TX: score=0.6175, income=89627, unemployment=2.90, est/10k=331.15
- Bozeman, MT: score=0.5970, income=103918, unemployment=2.30, est/10k=537.71
- San Francisco-Oakland-Fremont, CA: score=0.5773, income=135590, unemployment=4.20, est/10k=286.64
- Sioux Falls, SD-MN: score=0.5709, income=82509, unemployment=1.70, est/10k=297.04
- Bismarck, ND: score=0.5705, income=86769, unemployment=2.30, est/10k=299.77
- Casper, WY: score=0.5699, income=71381, unemployment=3.30, est/10k=381.92
- Bridgeport-Stamford-Danbury, CT: score=0.5698, income=116402, unemployment=3.10, est/10k=276.51
- Minot, ND: score=0.5598, income=70782, unemployment=2.50, est/10k=302.45
- Rapid City, SD: score=0.5557, income=78056, unemployment=1.80, est/10k=343.32

## Limitations
- CBP is 2022 while ACS and BLS are 2024, so the study has a small time mismatch.
- This is a metro-level market study, not a business-level causal model.
- The opportunity index is a decision-support ranking tool, not a single ground-truth outcome.
- Phase 3 recommendation strategies are rule-based extensions on top of the Phase 2 model outputs.