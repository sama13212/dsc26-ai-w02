# AI usage

## Tools I used, and for which functions

I used ChatGPT to help me understand the task requirements, check Python syntax, and run and interpret the public tests. I also used it to review the logic of functions in `audit.py` and `metrics.py`. To understand ROC AUC and how it is used to detect potential data leakage, I referred to this DataCamp article: https://www.datacamp.com/tutorial/auc.

## One place where the AI was wrong, incomplete, or not what the spec said

The initial explanation made me think that I needed to build a model and generate prediction scores myself. After checking the README, I noticed that no model training was required because the grader provides the data and scores. This helped me understand that my task was to implement and test the library functions, not train a model.

## The function I wrote without AI, and why

I wrote parts of the `audit()` function myself to practice implementing the data-quality checks and understand how the audit results are calculated. I used AI for guidance and syntax checking, then ran the public tests to check the implementation against the task requirements.