# Framing: Rawaj Pay mule detection

## Decision

The model helps Dina, Head of Fraud, decide which transfers should be flagged for review by the five fraud analysts. The team's daily review capacity needs to be established so that the model does not generate more alerts than they can handle. A false positive wastes analysts' time reviewing a legitimate transfer, while a false negative allows a mule transfer to go undetected and may cause financial loss. The exact daily capacity and numerical costs of these errors are not specified in the task, so they should be agreed upon before choosing the final threshold.

## Metric

We should choose the threshold based on the cost of false positives and false negatives. Expected cost helps compare thresholds by assigning a cost to each type of error. We should also monitor precision and recall to understand how many flagged transfers are actually mules and how many mule transfers are detected. Accuracy alone is not suitable because mule transfers may be rare, and a model could achieve high accuracy while missing most mule transfers.

## Baseline

Before using a model, the team should compare its performance with the current fraud-review process or an existing rule-based approach. The model should reduce the cost of missed mule transfers and false alarms while keeping the number of flagged cases within the analysts' review capacity. The baseline must be evaluated using the same error costs and review constraints.

## Classical ML or LLM

A classical machine-learning model is a better starting point for scoring structured transfer data because its scores can be evaluated against labels and used to choose a threshold. Simple rules can provide a baseline. An LLM is not the first choice for scoring every transfer because the task requires consistent, measurable decisions on structured data. The final choice should be judged by detection quality, error costs, and analyst capacity.