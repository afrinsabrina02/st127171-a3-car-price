 
import time
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------
# Task 2: regularization classes (same idea as the Ridge class in Regularization.ipynb)
# ----------------------------------------------------------------------------------
class Ridge:
    """L2 penalty:  lambda * sum(theta^2).

    The bias row (row 0 of W, because we prepend a column of ones to X) is NOT penalised.
    The assignment formula sums j from 1 to n, so theta_0 is left out.
    Shrinking the intercept would only push predictions toward a fixed value
    and does nothing to reduce the model's complexity.
    """

    def __init__(self, l):
        self.l = l

    def __call__(self, W):
        # penalty added to the loss; W shape (n, k), skip the bias row W[0]
        return self.l * np.sum(np.square(W[1:]))

    def derivation(self, W):
        # d/dW of lambda * W^2 = 2 * lambda * W ; bias row gets zero penalty gradient
        grad = 2 * self.l * W
        grad[0] = 0.0
        return grad


class NoPenalty:
    """Used when use_ridge=False, so the training loop never needs an if-statement."""

    def __call__(self, W):
        return 0.0

    def derivation(self, W):
        return np.zeros_like(W)


# ----------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------
class LogisticRegression:
    """Multinomial logistic regression (softmax) trained with gradient descent.

    Parameters
    ----------
    k          : number of classes
    n          : number of features INCLUDING the intercept column of ones
    method     : 'batch', 'minibatch' or 'sto'
    alpha      : learning rate
    max_iter   : number of parameter updates
    use_ridge  : if True, add the L2 penalty (Task 2)
    l          : lambda, the strength of the L2 penalty (ignored when use_ridge=False)
    batch_frac : fraction of the training rows used per mini-batch step
    verbose    : print the loss every 500 iterations
    """

    def __init__(self, k, n, method="batch", alpha=0.001, max_iter=5000,
                 use_ridge=False, l=0.01, batch_frac=0.3, verbose=True):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.use_ridge = use_ridge
        self.l = l
        self.batch_frac = batch_frac
        self.verbose = verbose
        # pick the penalty object once; both share the same interface
        self.regularization = Ridge(l) if use_ridge else NoPenalty()

    # ------------------------------------------------------------------ training
    def fit(self, X, Y):
        """X shape (m, n) with the intercept column, Y shape (m, k) one-hot."""
        self.W = np.random.rand(self.n, self.k)   # random start, same as the lecture
        self.losses = []
        m = X.shape[0]
        start_time = time.time()

        if self.method not in ("batch", "minibatch", "sto"):
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

        batch_size = max(1, int(self.batch_frac * m))

        for i in range(self.max_iter):
            if self.method == "batch":
                # every sample is used in every step
                X_b, Y_b = X, Y
            elif self.method == "minibatch":
                # a random subset of rows (no repeats inside one batch)
                ix = np.random.choice(m, size=batch_size, replace=False)
                X_b, Y_b = X[ix], Y[ix]
            else:  # 'sto'
                # exactly one random row per step; reshape keeps the 2-D shape (1, n) / (1, k)
                ix = np.random.randint(m)
                X_b, Y_b = X[ix:ix + 1], Y[ix:ix + 1]

            loss, grad = self.gradient(X_b, Y_b)
            self.losses.append(loss)
            self.W = self.W - self.alpha * grad   # gradient descent update

            if self.verbose and i % 500 == 0:
                print(f"Loss at iteration {i}", loss)

        self.train_time = time.time() - start_time
        if self.verbose:
            print(f"time taken: {self.train_time}")
        return self

    def gradient(self, X, Y):
        """Returns (loss, gradient) for the given rows.

        loss = -(1/m) * sum(Y * log(h))  +  lambda * sum(W[1:]^2)
        grad = (1/m) * X^T (h - Y)       +  2 * lambda * W  (bias row excluded)
        """
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        # clip so log never sees exactly 0
        cross_entropy = -np.sum(Y * np.log(np.clip(h, 1e-15, 1.0))) / m
        loss = cross_entropy + self.regularization(self.W)

        error = h - Y                                   # (m, k)
        grad = self.softmax_grad(X, error) / m          # (n, k), averaged like the loss
        grad = grad + self.regularization.derivation(self.W)
        return loss, grad

    def softmax(self, theta_t_x):
        # subtracting the row max does not change the result but avoids exp() overflow
        z = theta_t_x - np.max(theta_t_x, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        """X (m, n) @ W (n, k) -> probabilities (m, k), each row sums to 1."""
        return self.softmax(X @ W)

    # ------------------------------------------------------------------ prediction
    def predict_proba(self, X):
        return self.h_theta(X, self.W)

    def predict(self, X_test):
        """Class with the highest probability for every row -> shape (m,)."""
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("iteration")
        plt.ylabel("losses")
        plt.legend()

    # ------------------------------------------------------------------ Task 1: metrics
    # Every metric below works on class LABELS (shape (m,)), not one-hot.
    # Per class c:  TP = predicted c and really c
    #               FP = predicted c but really something else
    #               FN = really c but predicted something else

    def _counts(self, y_true, y_pred, c):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        return tp, fp, fn

    def accuracy(self, y_true, y_pred):
        """correct predictions / all predictions"""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        return np.sum(y_true == y_pred) / len(y_true)

    def precision(self, y_true, y_pred, c):
        """TP / (TP + FP) for class c. If the class is never predicted we return 0
        (the same convention as sklearn's default zero_division)."""
        tp, fp, _ = self._counts(y_true, y_pred, c)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, y_true, y_pred, c):
        """TP / (TP + FN) for class c. 0 if the class never appears in y_true."""
        tp, _, fn = self._counts(y_true, y_pred, c)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def f1_score(self, y_true, y_pred, c):
        """2 * precision * recall / (precision + recall) for class c."""
        p = self.precision(y_true, y_pred, c)
        r = self.recall(y_true, y_pred, c)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def support(self, y_true, c):
        """How many samples of class c are in the TRUE labels."""
        return int(np.sum(np.asarray(y_true) == c))

    # ---- macro averages: every class counts the same
    def macro_precision(self, y_true, y_pred):
        return np.mean([self.precision(y_true, y_pred, c) for c in range(self.k)])

    def macro_recall(self, y_true, y_pred):
        return np.mean([self.recall(y_true, y_pred, c) for c in range(self.k)])

    def macro_f1(self, y_true, y_pred):
        return np.mean([self.f1_score(y_true, y_pred, c) for c in range(self.k)])

    # ---- weighted averages: each class is weighted by its share of the true labels
    def _weights(self, y_true):
        sup = np.array([self.support(y_true, c) for c in range(self.k)])
        return sup / sup.sum()

    def weighted_precision(self, y_true, y_pred):
        w = self._weights(y_true)
        return np.sum(w * np.array([self.precision(y_true, y_pred, c) for c in range(self.k)]))

    def weighted_recall(self, y_true, y_pred):
        w = self._weights(y_true)
        return np.sum(w * np.array([self.recall(y_true, y_pred, c) for c in range(self.k)]))

    def weighted_f1(self, y_true, y_pred):
        w = self._weights(y_true)
        return np.sum(w * np.array([self.f1_score(y_true, y_pred, c) for c in range(self.k)]))

    def classification_report(self, y_true, y_pred):
        """Prints a report that looks like sklearn's classification_report and
        returns the same numbers as a dict so they can be compared or logged."""
        report = {}
        lines = [f"{'':>14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"]
        for c in range(self.k):
            p, r, f = (self.precision(y_true, y_pred, c),
                       self.recall(y_true, y_pred, c),
                       self.f1_score(y_true, y_pred, c))
            s = self.support(y_true, c)
            report[str(c)] = {"precision": p, "recall": r, "f1-score": f, "support": s}
            lines.append(f"{c:>14}{p:>10.2f}{r:>10.2f}{f:>10.2f}{s:>10}")

        total = len(y_true)
        acc = self.accuracy(y_true, y_pred)
        report["accuracy"] = acc
        lines.append("")
        lines.append(f"{'accuracy':>14}{'':>10}{'':>10}{acc:>10.2f}{total:>10}")

        macro = (self.macro_precision(y_true, y_pred), self.macro_recall(y_true, y_pred),
                 self.macro_f1(y_true, y_pred))
        weighted = (self.weighted_precision(y_true, y_pred), self.weighted_recall(y_true, y_pred),
                    self.weighted_f1(y_true, y_pred))
        report["macro avg"] = dict(zip(["precision", "recall", "f1-score"], macro), support=total)
        report["weighted avg"] = dict(zip(["precision", "recall", "f1-score"], weighted), support=total)
        lines.append(f"{'macro avg':>14}{macro[0]:>10.2f}{macro[1]:>10.2f}{macro[2]:>10.2f}{total:>10}")
        lines.append(f"{'weighted avg':>14}{weighted[0]:>10.2f}{weighted[1]:>10.2f}{weighted[2]:>10.2f}{total:>10}")
        print("\n".join(lines))
        return report


def one_hot(y, k):
    """Class labels (m,) -> one-hot matrix (m, k). Same idea as the lecture's for-loop,
    written as one line: row i of np.eye(k) has a 1 in column y[i]."""
    return np.eye(k)[np.asarray(y).astype(int)] 
