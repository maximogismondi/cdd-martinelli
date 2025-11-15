"""
Utilidades comunes para Machine Learning
Funciones reutilizables para evaluación y visualización de modelos
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Constantes de colores
COLOR_NO_DISASTER = "#3498db"
COLOR_DISASTER = "#e74c3c"
COLOR_GENERAL = "#95a5a6"


def evaluate_model(y_true, y_pred, model_name="Model", print_results=True):
    """
    Evalúa un modelo y muestra métricas + matriz de confusión

    Parameters
    ----------
    y_true : array-like
        Labels verdaderas
    y_pred : array-like
        Predicciones del modelo
    model_name : str
        Nombre del modelo para el título
    print_results : bool
        Si True, imprime métricas y muestra gráficos

    Returns
    -------
    dict
        Diccionario con métricas: f1, accuracy, precision, recall
    """
    if print_results:
        print("=" * 60)
        print(f"EVALUACIÓN: {model_name}".center(60))
        print("=" * 60)

    # Métricas
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    if print_results:
        print("\n📊 Métricas:")
        print(f"  F1 Score:  {f1:.4f}  ⭐ (métrica principal)")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disaster", "Disaster"],
            yticklabels=["No Disaster", "Disaster"],
            ax=ax,
            cbar_kws={"label": "Count"},
        )
        ax.set_xlabel("Predicción", fontsize=12, fontweight="bold")
        ax.set_ylabel("Real", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Matriz de Confusión - {model_name}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.show()

        # Classification report
        print("\n📋 Classification Report:")
        print(
            classification_report(
                y_true, y_pred, target_names=["No Disaster", "Disaster"]
            )
        )

    return {"f1": f1, "accuracy": acc, "precision": prec, "recall": rec}


def plot_feature_importance(feature_names, importance_values, model_name, top_n=20):
    """
    Grafica la importancia de features

    Parameters
    ----------
    feature_names : list
        Nombres de las features
    importance_values : array-like
        Valores de importancia
    model_name : str
        Nombre del modelo
    top_n : int
        Número de features más importantes a mostrar
    """
    # Crear DataFrame y ordenar
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)

    # Top features
    top_features = importance_df.head(top_n)

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        COLOR_DISASTER if imp > 0 else COLOR_NO_DISASTER
        for imp in top_features["importance"]
    ]

    ax.barh(range(len(top_features)), top_features["importance"], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Importancia", fontweight="bold")
    ax.set_title(
        f"Top {top_n} Features más Importantes - {model_name}",
        fontweight="bold",
        fontsize=14,
        pad=20,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return importance_df


def compare_models(results_dict):
    """
    Compara múltiples modelos en una tabla

    Parameters
    ----------
    results_dict : dict
        Diccionario con {nombre_modelo: dict_metricas}
    """
    df = pd.DataFrame(results_dict).T
    df = df[["f1", "accuracy", "precision", "recall"]]  # Ordenar columnas
    df = df.round(4)

    print("=" * 70)
    print("COMPARACIÓN DE MODELOS".center(70))
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    # Encontrar mejor modelo por F1
    best_model = df["f1"].idxmax()
    best_f1 = df.loc[best_model, "f1"]
    print(f"\n🏆 Mejor modelo (F1): {best_model} con F1={best_f1:.4f}")

    return df
