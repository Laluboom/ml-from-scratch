# Main Question

How can Kaggle datasets and competitions be used as a laboratory for learning data exploration, pattern discovery, feature analysis, and model-informed understanding across many types of real-world prediction problems?

## Reference Sub-Questions

### Exploratory Data Analysis as the Core Skill

- How can we inspect a new dataset quickly enough to understand its structure, target, feature types, missing values, and likely risks?
- What questions should be asked before modeling: what is each row, what is being predicted, where did the data come from, and what could be biased or missing?
- Which summaries reveal the most useful information: column distributions, target balance, grouped statistics, correlations, missingness maps, duplicates, and outlier checks?
- How can plots expose patterns that summary tables miss?
- How do numerical, categorical, text, date/time, geospatial, image, and high-cardinality features need different exploration methods?
- What makes an observation interesting enough to investigate further rather than just report?
- How can exploratory analysis separate real signal from noise, leakage, coincidence, or sampling artifacts?
- How should EDA findings be written so they become reusable insights instead of scattered notebook cells?

### Comparing Many Kaggle Datasets

- What recurring data patterns appear across competitions such as Titanic, House Prices, Spaceship Titanic, Tabular Playground, Store Sales, Home Credit, Porto Seguro, and NLP or image challenges?
- How do different domains change the analysis: survival data, housing prices, credit risk, sales forecasting, medical data, customer behavior, images, or text?
- Which exploration checklist stays the same across datasets, and which steps must change by problem type?
- How do classification, regression, forecasting, ranking, recommendation, NLP, and computer vision competitions require different analytical questions?
- What can beginner datasets teach well, and where do they become misleading compared with harder competitions?
- How can public Kaggle notebooks be used as references for analysis style, feature ideas, validation design, and visualization without copying blindly?
- What common mistakes appear repeatedly in Kaggle analyses: leakage, overinterpreting correlations, weak validation, bad missing-value handling, or leaderboard chasing?

### Feature Understanding and Data Signals

- Which features look predictive before modeling, and how can that be tested honestly?
- How can group comparisons, target-rate analysis, mutual information, correlation, chi-square tests, and simple baseline models reveal candidate signals?
- When does a missing value carry meaning rather than needing simple imputation?
- How can interactions between features reveal patterns that single-column analysis hides?
- How can domain knowledge turn raw columns into useful features such as ratios, counts, recency, frequency, aggregates, bins, or flags?
- How do target encoding, frequency encoding, one-hot encoding, embeddings, and native categorical handling change what a model can learn?
- Which engineered features are stable across folds, and which only look useful by chance?
- How can feature analysis detect leakage, proxy variables, duplicate records, or unrealistic shortcuts?

### Model-Informed Analysis

- How can simple models be used as analysis tools before trying to maximize leaderboard score?
- What can linear models, decision trees, random forests, gradient boosting, nearest neighbors, and neural networks reveal about the data differently?
- How can feature importance, permutation importance, SHAP values, partial dependence, residual analysis, and error slicing explain model behavior?
- How can model errors show where the dataset is difficult, noisy, underrepresented, or poorly described by available features?
- When does a model confirm an EDA hypothesis, and when does it expose that the hypothesis was weak?
- How can comparing several model families reveal whether the problem depends on linear effects, interactions, nonlinearity, rare categories, or complex representations?
- How can baseline models help decide whether more data cleaning, feature engineering, or different validation matters more than algorithm choice?
- How should model explanations be checked against the raw data so they do not become false stories?

### Validation for Analysis Quality

- How can validation be used to test whether discovered patterns generalize beyond the visible training data?
- What split best matches the problem structure: random, stratified, group-based, time-based, cross-validation, or nested validation?
- How can validation expose unstable features, leakage, public leaderboard overfitting, and distribution shift?
- What should be logged for each analysis experiment: data version, hypothesis, features tested, model used, validation method, score, and conclusion?
- How much score movement is meaningful, and how much is normal variance?
- How can subgroup validation show that a model works for some cases but fails for others?
- How can public and private leaderboard differences be used as lessons about generalization rather than just ranking?

### Visualization and Communication

- Which visualizations best explain different data questions: histograms, box plots, scatter plots, heatmaps, bar charts, line plots, pair plots, maps, confusion matrices, and residual plots?
- How can visualizations be designed to answer a specific question instead of decorating a notebook?
- How should uncertainty, sample size, missingness, class imbalance, and outliers be shown clearly?
- How can a notebook tell a story from data understanding to modeling insight without hiding failed experiments?
- What should a final analysis report include: problem framing, dataset summary, key findings, model evidence, limitations, and next experiments?
- How can analysis be written for both technical readers and non-technical stakeholders?

### Experiments and Testbeds

- Which Kaggle competitions should be used as reference testbeds for specific skills: Titanic for categorical EDA, House Prices for regression, Store Sales for forecasting, Home Credit for messy tabular data, and image or NLP competitions for unstructured data?
- How can we create a reusable EDA template and test it across several competitions?
- How can we build a feature-analysis template that records each hypothesis and whether validation supports it?
- How can we compare the same modeling approach across multiple datasets to learn when it succeeds or fails?
- How can we build a library of examples showing leakage, missing-value signal, useful interactions, unstable features, and validation mistakes?
- What small experiments can prove that a visualization, feature, or model explanation actually improves understanding?

## Current Project Materials

- `kaggle_titanic/`: placeholder for a Titanic-style beginner Kaggle analysis project.
- `prediction_modeling/`: placeholder for general prediction-modeling and analysis experiments.
- `wiki/`: space for notes on datasets, EDA methods, validation lessons, model explanations, and competition comparisons.
- `research_reports/run_logs/`: space for dated analysis logs, experiment notes, and findings from Kaggle testbeds.
