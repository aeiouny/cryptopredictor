# MVP Notes

## Coin
Bitcoin

## Anomaly Rule
An anomaly is detected when the absolute z-score of returns is greater than 2.5.

## Breakout Label
After the anomaly, price continues in the same direction by at least 1% within 30 minutes and does not quickly return to the pre-anomaly level.

## Reversal Label
After the anomaly, price moves back toward the pre-anomaly level or changes direction within 30 minutes.

## First Model
Logistic Regression

## Success Criteria
- historical data collected
- anomaly events detected
- breakout vs reversal labels generated
- baseline model trained
- predictions stored in AWS