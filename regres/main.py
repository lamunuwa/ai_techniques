import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RegresionLinealNumPy:
    """Regresión lineal entrenada mediante descenso de gradiente."""

    def __init__(self, lr: float = 0.05, n_iter: int = 1000) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegresionLinealNumPy":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iter):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            dw = (2.0 / n_samples) * (X.T @ error)
            db = (2.0 / n_samples) * float(np.sum(error))
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("El modelo debe entrenarse antes de predecir.")
        return X @ self.weights + self.bias


class RegresionLogisticaNumPy:
    """Regresión logística binaria entrenada mediante descenso de gradiente."""

    def __init__(self, lr: float = 0.1, n_iter: int = 1000) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegresionLogisticaNumPy":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iter):
            linear = X @ self.weights + self.bias
            probs = self._sigmoid(linear)
            error = probs - y
            dw = (1.0 / n_samples) * (X.T @ error)
            db = (1.0 / n_samples) * float(np.sum(error))
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("El modelo debe entrenarse antes de predecir.")
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def ejecutar_regresion_lineal() -> None:
    """Compara regresión lineal en NumPy vs. scikit-learn."""
    X, y = make_regression(
        n_samples=300,
        n_features=3,
        noise=15.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo_np = RegresionLinealNumPy(lr=0.05, n_iter=1000).fit(X_train_scaled, y_train)
    pred_np = modelo_np.predict(X_test_scaled)

    modelo_sk = LinearRegression().fit(X_train_scaled, y_train)
    pred_sk = modelo_sk.predict(X_test_scaled)

    print("=== Regresión Lineal ===")
    for nombre, pred in [("NumPy", pred_np), ("scikit-learn", pred_sk)]:
        mse = mean_squared_error(y_test, pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, pred)
        print(f"{nombre:12s} | MSE: {mse:8.3f} | RMSE: {rmse:7.3f} | R2: {r2:.4f}")


def ejecutar_regresion_logistica() -> None:
    """Compara regresión logística en NumPy vs. scikit-learn."""
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo_np = RegresionLogisticaNumPy(lr=0.1, n_iter=1000).fit(
        X_train_scaled, y_train
    )
    pred_np = modelo_np.predict(X_test_scaled)

    modelo_sk = LogisticRegression(max_iter=1000, random_state=42).fit(
        X_train_scaled, y_train
    )
    pred_sk = modelo_sk.predict(X_test_scaled)

    print("\n=== Regresión Logística ===")
    for nombre, pred in [("NumPy", pred_np), ("scikit-learn", pred_sk)]:
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(
            f"{nombre:12s} | Acc: {acc:.4f} | Prec: {prec:.4f} "
            f"| Rec: {rec:.4f} | F1: {f1:.4f}"
        )


def main() -> None:
    ejecutar_regresion_lineal()
    ejecutar_regresion_logistica()


if __name__ == "__main__":
    main()
