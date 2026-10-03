# A3: Predicting Car Price (III) - Classification

Student: st127171 | Course: AT82.03 Machine Learning

The Car Price dataset is treated as a **4-class classification** problem: the selling price is put into 4 quartile buckets
(class 0 = cheapest quarter ... class 3 = most expensive quarter) and predicted with a **multinomial logistic regression written from scratch**.

| Class | Price range |
|---|---|
| 0 | up to 270,000 |
| 1 | 270,000 to 450,000 |
| 2 | 450,000 to 690,000 |
| 3 | above 690,000 |

## Contents
| Path | What it is |
|---|---|
| `A3_Predicting_Car_Price.ipynb` | The whole assignment: preprocessing, Task 1 (metrics), Task 2 (ridge), Task 3 (MLflow) |
| `logistic_regression.py` | The model class (softmax regression with batch / mini-batch / stochastic gradient descent, ridge option, from-scratch metrics) |
| `tests/test_model.py` | 2 unit tests (expected input, expected output shape) |
| `.github/workflows/ci-cd.yml` | GitHub Actions: run the tests on every push, deploy only if they pass |
| `app/` | Dash web app, Dockerfile, requirements and the exported model (`app/model/car_model.pkl`) |

## Task 1 - Classification metrics from scratch
Implemented inside the model class: `accuracy`, per-class `precision`, `recall`, `f1_score`, `support`, the `macro_*` and `weighted_*`
averages and a `classification_report`. They were compared with scikit-learn's `classification_report` on mock-up data and on the
real predictions, and the numbers match. *Support* is the number of samples of a class in the true labels.

## Task 2 - Ridge logistic regression
`use_ridge=True, l=<lambda>` adds `lambda * sum(W^2)` to the loss (the bias is not penalised).
Same split, same start weights, batch gradient descent, alpha = 0.1, 2000 iterations:

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| No ridge | 74.46% | 0.745 |
| Ridge (lambda = 0.01) | 71.49% | 0.71 |

Ridge did not help here: with about 6,300 training rows and 47 features the model does not overfit much.

## Task 3 - MLflow, model registry, CI/CD
**Experiments.** Experiment `st127171-a3`, 24 runs (3 gradient descent methods x no ridge / 3 lambdas x 2 learning rates).
Parameters, validation and test metrics, a loss curve and the saved model are logged for each run. The dataset is not logged.
Runs are compared on a validation split cut from the training data, so the test set stays untouched.

*Note:* the experiments were logged to a **local MLflow store** (`sqlite:///mlflow.db`) because the CSIM MLflow server refused
connections from my network (connection refused on ports 80 and 443; reported to the TA). To log to the server, only `TRACKING_URI`
in the notebook has to change.

**Findings.**
* Best run (highest validation macro-F1): `batch-alpha0.1-noridge`, validation macro-F1 0.7248, test accuracy 0.7402.
* The learning rate matters most: alpha = 0.1 gives about 0.72 validation macro-F1, alpha = 0.01 only about 0.65 (not converged in 2000 iterations).
* Ridge with lambda = 0.001 is the same as no ridge; lambda = 0.01 costs about 3 points.
* Batch and mini-batch behave almost identically. The top 4 runs are within noise of each other.

**Model registry.** The best run is registered as `st127171-a3-model` (version 1) with the `staging` alias.

**CI/CD.** Every push runs the two unit tests (`python -m pytest`). The deploy job builds the Docker image of `app/` and starts it on
the course virtual machine, but only if the tests passed.

## Run it locally
```bash
pip install -r app/requirements.txt
python app/app.py           # open http://localhost:8050
python -m pytest -v         # unit tests (needs: pip install numpy matplotlib pytest)
```

## GitHub secrets used by the deploy job
`DOCKER_USERNAME`, `DOCKER_PASSWORD`, `SSH_HOST`, `SSH_PORT`, `SSH_USERNAME`, `SSH_KEY`


## Deployment note
The deploy job is configured but switched off (it only runs when the repository variable DEPLOY_ENABLED is set to true), because the Docker Hub login and the course VM details were not available. The test job runs on every push and passes.
