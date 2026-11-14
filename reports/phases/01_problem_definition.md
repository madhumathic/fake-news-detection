# Phase 1 - Problem Definition

## Background
Online news spreads quickly and much of it is false or misleading. Readers cannot verify every story, so tools that flag
articles that *look like* fake news can support, but never replace, human judgement.

## Problem statement
Given the text of a news article (headline and body), predict whether it is **REAL NEWS** or **FAKE NEWS**.

## Existing problem
Manual fact-checking is slow and cannot keep up with the volume of published articles. Simple keyword rules are easy to
evade and produce many false alarms.

## Proposed solution
A supervised machine-learning classifier: article text is converted to TF-IDF features and a linear classifier trained on a
labelled dataset predicts the class. The model is served through a web application with a prediction history.

## Main objective
Build a complete, reproducible AI system (data to deployed web app) that classifies news articles with measured, honestly
reported accuracy.

## Specific objectives
1. Load, validate and clean the supplied Kaggle dataset (news_dataset.csv).
2. Explore the data and check it for data leakage.
3. Extract TF-IDF features and compare several algorithms fairly.
4. Tune and train the best model, then evaluate it once on unseen data.
5. Save the model and vectorizer, and integrate them into a web application with SQLite history.
6. Test the system and document every phase.

## Scope
**In scope:** English news articles, binary classification (REAL / FAKE), local web application, prediction history.
**Out of scope:** verifying individual claims against external sources, images or video, other languages, real-time
monitoring of social media.

## Users and stakeholders
Students and examiners (demonstration and learning), general readers who want a second opinion on an article's style.

## Success criteria
* The model beats a majority-class baseline with statistical significance on unseen test data.
* No identified data-leakage feature is used for prediction.
* Every pipeline step is reproducible from the command line; all automated tests pass.

## Limitations and assumptions
The system predicts from writing style and vocabulary learned from the training dataset. It does not check facts, and its
accuracy applies to articles similar to the training data.
