# Premier League Match Simulation Tool

## Project Status: COMPLETE

### Introduction
This project uses a large dataset of football match games from https://beatthebookie.blog/ to model xG in games using prior information relating to team and opponent strength as features.

It uses exponential moving averages (EMAs) of statsitics such as xG, shots, deep completions and more. 

Once the xG predictions and the model were selected, the model's xG predictions are used to generated match odds, and then a poission model to predict match outcomes is used - this is then compared against the fair odds from bet365 (margin removed) to evaluate model performance using a Brier score.

### Conclusions
My XGBoost model slightly outperforms the Bet365 odds, when removing the margin, to predict the outcome of football matches. The EMA of "shots for" is the most important feature when using this model, with the home effect being the least important.