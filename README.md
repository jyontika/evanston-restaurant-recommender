# 🍽️ Evanston Restaurant Recommender

An end-to-end recommendation system built on real restaurant review data from Evanston, IL. This project demonstrates how modern recommendation techniques — from popularity scoring to NLP-powered sentiment modeling — can be applied to drive user engagement and increase conversion in a review or dining platform.

## Business motivation

Recommender systems are a core driver of revenue in consumer platforms. Amazon attributes ~35% of revenue to recommendations; Netflix credits them with $1B+ in annual retention value. This project applies the same methods — collaborative filtering, matrix factorization, and text embeddings — to a restaurant review dataset, showing how similar infrastructure can improve user experience and reduce churn in food-tech or local discovery applications.

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
└── data/
    └── RestaurantReviews.xlsx
```

## Author

Jyontika Kapoor · [jyontika@u.northwestern.edu](mailto:jyontika@u.northwestern.edu)
