# Introduction to Machine Learning

Machine learning is a branch of artificial intelligence in which computers learn patterns from data instead of following hand-written rules. A model is trained on examples, and the trained model is then used to make predictions on new, unseen data.

## Types of machine learning

**Supervised learning** uses labeled examples. Each training example has an input and the correct output, for instance an email and the label "spam" or "not spam". Common supervised tasks are classification, which predicts a category, and regression, which predicts a number such as a house price.

**Unsupervised learning** works with unlabeled data. The goal is to discover structure, such as grouping similar customers together with clustering or reducing the number of features with dimensionality reduction.

**Reinforcement learning** trains an agent that takes actions in an environment and receives rewards or penalties. Over time the agent learns a policy that maximizes the total reward. Game-playing systems and robot controllers often use this approach.

## Training, validation and test data

A dataset is usually split into three parts. The training set, commonly about 70 percent of the data, is used to fit the model. The validation set, about 15 percent, is used to tune hyperparameters and to choose between models. The test set, about 15 percent, is held back until the end to give an honest estimate of how the model will perform on new data.

## Overfitting and underfitting

Overfitting happens when a model memorizes the training data, including its noise, and therefore performs poorly on new data. Underfitting happens when a model is too simple to capture the real pattern. Techniques that reduce overfitting include collecting more data, using regularization, applying dropout in neural networks, and stopping training early when the validation error starts to rise.

## Evaluation metrics

For classification, accuracy is the share of correct predictions. Precision is the share of predicted positives that are truly positive, and recall is the share of actual positives that the model finds. The F1 score is the harmonic mean of precision and recall. For regression, common metrics are mean absolute error (MAE) and root mean squared error (RMSE).
