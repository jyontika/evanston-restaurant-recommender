# 🍽️ Evanston Restaurant Recommender

An end-to-end recommendation system built on restaurant review data from Evanston, IL to help users discover restaurants aligned with their preferences. The project combines popularity-based recommendation, sentiment modeling, and review analysis to show how data-driven ranking systems can improve user experience on dining or review platforms.

## Business motivation

Recommender systems are a core driver of revenue in consumer platforms. They help users identify options they are more likely to enjoy from large catalogs of options, reducing decision fatigue and increasing engagement. Commonly cited estimates suggest that Amazon attributes roughly 35% of purchases to recommendations, while Netflix has valued its recommendation system at roughly $1B annually through improved subscriber retention.

In this project, I apply the same ideas to a smaller, local setting: restaurant reviews from Evanston, IL. Using content-based filtering, collaborative filtering, and NLP-based sentiment modeling, the system recommends restaurants based on user review patterns and restaurant characteristics. The broader motivation is to understand how recommendation infrastructure can help users discover places they are likely to enjoy, while improving engagement and retention for food-tech or local discovery platforms.

## Data
Restaurant review data from Evanston, IL (68 restaurants, 1,500 reviews). Sourced from a course project at Northwestern University (STAT 415, Spring 2025). Raw data is not included in this repo.

## Methods

### Popularity-based ranking
- Bayesian shrinkage estimator to balance rating quality against review volume — directly addressing the cold-start problem for new restaurants
- Cuisine-specific popularity engine for targeted recommendations

### Content-based filtering
- Restaurant similarity via Euclidean and cosine distance on one-hot encoded features
- Jaccard similarity on augmented NLP descriptions for vocabulary-overlap matching
- Systematic comparison of distance metrics using cuisine match rate and cost proximity as evaluation criteria

### Collaborative filtering
- Demographic-based user similarity using cosine distance on 14-dimensional feature vectors
- SVD matrix factorization (via `surprise`) to impute sparse user–item rating matrices — the standard approach used in production recommenders
- Review-based user similarity with deviation-weighted scoring to surface genuinely enthusiastic ratings

### Predictive modeling
- Linear and logistic regression baselines with SelectKBest feature selection
- L1 (Lasso) regularization for automatic feature selection — cuisine type dominates over demographics
- Sentence transformer embeddings (via `sentence-transformers`) on review text, achieving the lowest MSE (0.181) across all models — a **42% improvement** over the logistic regression baseline

## Key results

| Model | Metric | Value |
|---|---|---|
| Sentence transformers + logistic regression | MSE | 0.181 |
| Logistic regression (SMOTE + threshold) | MSE | 0.315 |
| Linear regression | MSE | 1.915 |
| SVD collaborative filter | RMSE (5-fold CV) | 1.309 |

- Demographic features alone are weak predictors; cuisine preference and review text carry the signal
- Shrinkage estimator corrects inflated ratings from low-volume restaurants, improving ranking reliability
- L1 regularization confirms cuisine type as the dominant feature; demographic weights are near zero

## Tech stack

`Python` · `pandas` · `scikit-learn` · `sentence-transformers` · `surprise` · `imbalanced-learn (SMOTE)` · `matplotlib` · `seaborn`

## Repo structure

```
├── notebook_part1.ipynb   # EDA, popularity matching, content-based filtering, NLP similarity
├── notebook_part2.ipynb   # Collaborative filtering, SVD, regression models, text embeddings

```

## Author

Jyontika Kapoor · [jyontika@u.northwestern.edu](mailto:jyontika@u.northwestern.edu)
