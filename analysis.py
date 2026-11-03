"""
Предсказание оттока клиентов телеком-компании (Telco Customer Churn).

Скрипт делает весь проект от начала до конца:
  1. Исследование данных (EDA)
  2. SQL-запросы к данным (SQLite в памяти)
  3. Подготовка данных и разделение на train/test
  4. Обучение моделей: базовый ориентир, логистическая регрессия, дерево, случайный лес
  5. Метрики на тестовой выборке
  6. Интерпретация: коэффициенты и важность признаков
  7. Графики в images/
  8. Самопроверка расчётов (assert'ы)

Запуск:  python analysis.py
Все числа печатаются в консоль и сохраняются в results/.
"""

from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, без окна
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SQL_DIR = ROOT / "sql"
IMG_DIR = ROOT / "images"
RES_DIR = ROOT / "results"

RANDOM_STATE = 42  # фиксируем случайность, чтобы результаты повторялись при каждом запуске
TEST_SIZE = 0.2

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

# Цвета для графиков: нейтральная поверхность, тёмный текст, проверенная палитра серий
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Исследование данных
# ---------------------------------------------------------------------------
def explore(raw):
    section("1. ИССЛЕДОВАНИЕ ДАННЫХ")
    print(f"Размер: {raw.shape[0]} строк, {raw.shape[1]} столбцов")
    print("\nТипы столбцов:")
    print(raw.dtypes.astype(str).value_counts().to_string())

    print(f"\nЯвных пропусков (NaN): {int(raw.isna().sum().sum())}")

    # TotalCharges прочитался как текст: в нём есть строки из пробелов.
    # pd.to_numeric(errors="coerce") превращает всё, что не число, в NaN — так их и найдём.
    total_num = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    blank = total_num.isna()
    print(f"TotalCharges хранится как текст ({raw['TotalCharges'].dtype}); "
          f"нечисловых (пустых) значений: {int(blank.sum())}")
    print(f"tenure у этих клиентов: {sorted(raw.loc[blank, 'tenure'].unique().tolist())} "
          f"— это новые клиенты, им ещё ни разу не выставили счёт")

    churn_share = (raw["Churn"] == "Yes").mean()
    print(f"\nДоля ушедших (Churn = Yes): {churn_share:.1%}")
    print(f"Доля оставшихся (Churn = No): {1 - churn_share:.1%}")
    print("Классы несбалансированы: оставшихся почти в 3 раза больше, чем ушедших.")

    print("\nЧисловые признаки:")
    print(pd.DataFrame({"tenure": raw["tenure"], "MonthlyCharges": raw["MonthlyCharges"],
                        "TotalCharges": total_num}).describe().round(2).to_string())

    return {
        "rows": int(raw.shape[0]),
        "cols": int(raw.shape[1]),
        "nan_total": int(raw.isna().sum().sum()),
        "total_charges_blank": int(blank.sum()),
        "churn_share": float(churn_share),
        "churned": int((raw["Churn"] == "Yes").sum()),
    }


# ---------------------------------------------------------------------------
# 2. SQL-запросы
# ---------------------------------------------------------------------------
def run_sql(raw):
    section("2. SQL-ЗАПРОСЫ (SQLite)")
    # Загружаем таблицу в SQLite в оперативной памяти и выполняем запросы из папки sql/
    con = sqlite3.connect(":memory:")
    raw.to_sql("customers", con, index=False)

    results = {}
    for path in sorted(SQL_DIR.glob("*.sql")):
        query = path.read_text(encoding="utf-8")
        table = pd.read_sql_query(query, con)
        results[path.stem] = table
        table.to_csv(RES_DIR / f"sql_{path.stem}.csv", index=False)
        print(f"\n--- {path.name} ---")
        print(table.to_string(index=False))
    con.close()
    return results


# ---------------------------------------------------------------------------
# 3. Подготовка данных
# ---------------------------------------------------------------------------
def clean(raw):
    df = raw.copy()
    # Пустой TotalCharges только у клиентов с tenure = 0: они ещё ничего не заплатили,
    # поэтому логично поставить 0. Это правило, а не статистика по данным
    # (не среднее и не медиана), поэтому утечки из теста в train здесь нет.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # SeniorCitizen закодирован 0/1, остальные бинарные признаки — Yes/No.
    # Приводим к одному виду, чтобы кодировать все категории одинаково.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    y = (df["Churn"] == "Yes").astype(int)  # 1 = ушёл, 0 = остался
    # customerID — просто номер, он не несёт информации о поведении клиента
    X = df.drop(columns=["customerID", "Churn"])
    return X, y


def make_preprocessor(categorical):
    # Числа масштабируем (среднее 0, стандартное отклонение 1), чтобы логистическая
    # регрессия могла честно сравнивать признаки разного масштаба.
    # Категории превращаем в столбцы 0/1 (one-hot). Для бинарных Yes/No оставляем
    # один столбец, а не два.
    # Важно: всё это «учится» (среднее, список категорий) только на train внутри Pipeline.
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
    ])


# ---------------------------------------------------------------------------
# 4. Модели
# ---------------------------------------------------------------------------
def build_models(categorical):
    # Каждая модель — это Pipeline «подготовка -> модель». Когда мы вызываем fit на train,
    # подготовка тоже обучается только на train. На test она лишь применяется.
    models = {
        # Базовый ориентир: всегда отвечает самым частым классом («никто не уйдёт»).
        # Любая полезная модель должна быть лучше него.
        "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
        # Логистическая регрессия: взвешенная сумма признаков -> вероятность через сигмоиду.
        "Логистическая регрессия": LogisticRegression(max_iter=1000),
        # Дерево решений: цепочка вопросов «да/нет». Глубину ограничиваем,
        # чтобы дерево не «зазубрило» обучающую выборку и оставалось читаемым.
        "Дерево решений": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                                                 random_state=RANDOM_STATE),
        # Случайный лес: много деревьев на случайных подвыборках, ответ — голосование.
        "Случайный лес": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }
    return {name: Pipeline([("prep", make_preprocessor(categorical)), ("model", m)])
            for name, m in models.items()}


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, categorical):
    section("4-5. ОБУЧЕНИЕ МОДЕЛЕЙ И МЕТРИКИ")
    models = build_models(categorical)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, probas = [], {}
    for name, pipe in models.items():
        # Кросс-валидация только на train: 5 раз учимся на 4/5 train и проверяемся на 1/5.
        # Это показывает, насколько стабильно качество, не трогая тестовую выборку.
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]  # вероятность ухода для каждого клиента
        probas[name] = proba
        m = evaluate(y_test, proba)
        m.update({"model": name, "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std()})
        rows.append(m)

    metrics = pd.DataFrame(rows).set_index("model")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean", "cv_roc_auc_std"]
    print("\nМетрики на тестовой выборке (порог 0.5) + ROC-AUC на кросс-валидации по train:")
    print(metrics[cols].round(3).to_string())

    print("\nМатрицы ошибок на test (строки — факт, столбцы — прогноз):")
    for name, r in metrics[["tn", "fp", "fn", "tp"]].iterrows():
        print(f"\n{name}")
        print(f"                 прогноз: остался   прогноз: ушёл")
        print(f"  факт: остался  {r['tn']:>15}   {r['fp']:>13}")
        print(f"  факт: ушёл     {r['fn']:>15}   {r['tp']:>13}")

    metrics.to_csv(RES_DIR / "metrics.csv")
    return models, metrics, probas


def example_customer(models, columns):
    # Придуманный клиент для объяснения моделей в README: «Анна».
    # Помесячный контракт, 3 месяца с компанией, оптоволокно, никаких доп. услуг.
    section("ПРИМЕР: КАК МОДЕЛИ ОЦЕНИВАЮТ ОДНОГО КЛИЕНТА")
    anna = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0, "TotalCharges": 225.0,
    }])[columns]
    probs = {name: float(pipe.predict_proba(anna)[0, 1]) for name, pipe in models.items()}
    for name, p in probs.items():
        print(f"{name:<25} вероятность ухода = {p:.2f}")

    # Путь Анны по дереву: какие вопросы ей задали и в какой лист она попала
    tree = models["Дерево решений"]
    est = tree.named_steps["model"]
    x = tree.named_steps["prep"].transform(anna)
    x = x.toarray() if hasattr(x, "toarray") else x
    names = [n.split("__", 1)[1] for n in tree.named_steps["prep"].get_feature_names_out()]
    node_ids = est.decision_path(x).indices
    print("\nПуть Анны по дереву решений:")
    for node in node_ids:
        if est.tree_.children_left[node] == -1:  # лист
            counts = est.tree_.value[node][0]
            share = counts[1] / counts.sum()  # value хранит доли классов в листе
            n = est.tree_.n_node_samples[node]
            print(f"  лист: {n} клиентов из train, доля ушедших {share:.1%}")
        else:
            f, thr = est.tree_.feature[node], est.tree_.threshold[node]
            raw_value = x[0, f]
            print(f"  {names[f]} <= {thr:.2f}?  у Анны {raw_value:.2f} -> "
                  f"{'да' if raw_value <= thr else 'нет'}")
    return probs


def threshold_demo(y_test, proba):
    # Порог 0.5 — не закон. Если бизнесу важнее найти побольше уходящих
    # (и не страшно позвонить лишним), порог можно снизить.
    # Это иллюстрация компромисса precision/recall; честно выбирать порог нужно
    # на отдельной валидационной выборке, а не на test.
    section("ИЛЛЮСТРАЦИЯ: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ПРИ РАЗНЫХ ПОРОГАХ")
    rows = []
    for t in [0.5, 0.3]:
        m = evaluate(y_test, proba, threshold=t)
        m["threshold"] = t
        m["flagged"] = m["tp"] + m["fp"]
        rows.append(m)
    table = pd.DataFrame(rows).set_index("threshold")
    print(table[["precision", "recall", "f1", "flagged", "tp", "fp", "fn"]].round(3).to_string())
    table.to_csv(RES_DIR / "logreg_thresholds.csv")
    return table


# ---------------------------------------------------------------------------
# 6. Интерпретация
# ---------------------------------------------------------------------------
def interpret(models, X_train, X_test, y_test):
    section("6. ИНТЕРПРЕТАЦИЯ")

    # 6a. Коэффициенты логистической регрессии.
    # Положительный коэффициент — признак повышает шанс ухода, отрицательный — снижает.
    # exp(коэф) = отношение шансов (odds ratio): во сколько раз меняются шансы ухода,
    # когда бинарный признак = 1 (или числовой растёт на 1 стандартное отклонение).
    logreg = models["Логистическая регрессия"]
    names = logreg.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({
        "feature": [n.split("__", 1)[1] for n in names],
        "coef": logreg.named_steps["model"].coef_[0],
    })
    coefs["odds_ratio"] = np.exp(coefs["coef"])
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
    coefs.to_csv(RES_DIR / "logreg_coefficients.csv", index=False)
    print("\nЛогистическая регрессия — 12 самых сильных коэффициентов:")
    print(coefs.head(12).round(3).to_string(index=False))
    # Для категорий понятнее сравнивать два значения одного признака между собой:
    # exp(coef_A - coef_B) = во сколько раз шансы ухода у A выше, чем у B,
    # при прочих равных признаках клиента.
    c = coefs.set_index("feature")["coef"]
    pairs = [
        ("Contract_Month-to-month", "Contract_Two year"),
        ("Contract_Month-to-month", "Contract_One year"),
        ("InternetService_Fiber optic", "InternetService_DSL"),
        ("PaymentMethod_Electronic check", "PaymentMethod_Credit card (automatic)"),
        ("PaymentMethod_Electronic check", "PaymentMethod_Bank transfer (automatic)"),
        ("OnlineSecurity_No", "OnlineSecurity_Yes"),
        ("TechSupport_No", "TechSupport_Yes"),
    ]
    pair_or = pd.DataFrame([{"A": a, "B": b, "odds_ratio_A_vs_B": np.exp(c[a] - c[b])}
                            for a, b in pairs])
    pair_or.to_csv(RES_DIR / "logreg_pairwise_odds_ratios.csv", index=False)
    print("\nОтношение шансов ухода A против B (при прочих равных):")
    print(pair_or.round(2).to_string(index=False))

    stds = X_train[NUMERIC].std()
    print("\n1 стандартное отклонение числовых признаков (на train): "
          + ", ".join(f"{c} = {stds[c]:.1f}" for c in NUMERIC))

    # 6b. Как выглядит дерево (первые уровни) — видно, какие вопросы оно задаёт.
    tree = models["Дерево решений"]
    tree_names = [n.split("__", 1)[1] for n in tree.named_steps["prep"].get_feature_names_out()]
    print("\nДерево решений — первые 3 уровня вопросов (числовые признаки в стандартизованном виде):")
    print(export_text(tree.named_steps["model"], feature_names=tree_names, max_depth=2))

    # 6c. Permutation importance: перемешиваем один столбец на test и смотрим,
    # насколько упал ROC-AUC. Сильно упал — модель на этот признак сильно опирается.
    # Считается по исходным столбцам (Contract целиком, а не Contract_One year отдельно).
    importances = {}
    for name in ["Логистическая регрессия", "Случайный лес"]:
        r = permutation_importance(models[name], X_test, y_test, scoring="roc_auc",
                                   n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        importances[name] = pd.Series(r.importances_mean, index=X_test.columns)
    imp = pd.DataFrame(importances).sort_values("Случайный лес", ascending=False)
    imp.to_csv(RES_DIR / "permutation_importance.csv")
    print("\nPermutation importance (падение ROC-AUC на test при перемешивании признака):")
    print(imp.round(4).to_string())
    return coefs, imp


# ---------------------------------------------------------------------------
# 7. Графики
# ---------------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)


def plot_contract(sql_results):
    t = sql_results["01_churn_by_contract"]
    order = ["Month-to-month", "One year", "Two year"]
    t = t.set_index("contract").loc[order]
    labels = [f"{name}\n{n} клиентов"
              for name, n in zip(["Помесячный", "1 год", "2 года"], t["customers"])]

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    style_axes(ax)
    bars = ax.bar(labels, t["churn_rate_pct"], color=SERIES[0], width=0.55)
    for bar, rate in zip(bars, t["churn_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=11)
    ax.set_ylim(0, max(t["churn_rate_pct"]) * 1.2)
    ax.set_ylabel("Доля ушедших, %")
    ax.set_title("Отток по типу контракта", loc="left", fontsize=13)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "churn_by_contract.png", dpi=150)
    plt.close(fig)


def plot_roc(y_test, probas, metrics):
    fig, ax = plt.subplots(figsize=(6.2, 5.6), facecolor=SURFACE)
    style_axes(ax)
    colors = iter(SERIES)
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        label = f"{name} (AUC = {metrics.loc[name, 'roc_auc']:.3f})"
        if name.startswith("Dummy"):
            # ROC-кривая Dummy совпадает с диагональю — это и есть «случайное угадывание»
            ax.plot(fpr, tpr, linestyle="--", color=TEXT_2, linewidth=1,
                    label=label + " = случайное угадывание")
        else:
            ax.plot(fpr, tpr, color=next(colors), linewidth=2, label=label)
    ax.set_xlabel("Доля ложных тревог (FPR)")
    ax.set_ylabel("Доля найденных уходящих (TPR = recall)")
    ax.set_title("ROC-кривые на тестовой выборке", loc="left", fontsize=13)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", fontsize=9, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_importance(imp, top=10):
    # Берём признаки, важные хотя бы для одной из моделей, самые важные — сверху
    top_imp = imp.loc[imp.max(axis=1).sort_values(ascending=False).index].head(top).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes, ["Случайный лес", "Логистическая регрессия"]):
        style_axes(ax)
        ax.barh(top_imp.index, top_imp[name], color=SERIES[0], height=0.6)
        ax.set_title(name, loc="left", fontsize=12)
        ax.set_xlabel("Падение ROC-AUC при перемешивании (шкалы разные)")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"Важность признаков (permutation importance, топ-{top})",
                 x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Самопроверка
# ---------------------------------------------------------------------------
def self_check(raw, X, y, X_train, X_test, y_train, y_test, sql_results, metrics):
    section("8. САМОПРОВЕРКА РАСЧЁТОВ")
    # SQL и pandas должны давать одинаковый отток по контрактам
    pd_rates = (raw.assign(c=raw["Churn"] == "Yes").groupby("Contract")["c"].mean() * 100).round(1)
    sql_rates = sql_results["01_churn_by_contract"].set_index("contract")["churn_rate_pct"]
    assert (pd_rates.sort_index() == sql_rates.sort_index()).all(), "SQL != pandas по контрактам"

    # Сумма клиентов в каждом SQL-запросе = все клиенты
    for name, t in sql_results.items():
        assert t["customers"].sum() == len(raw), f"{name}: потеряны строки"

    # После очистки не осталось пропусков, а train/test не пересекаются
    assert X.isna().sum().sum() == 0, "остались пропуски"
    assert len(set(X_train.index) & set(X_test.index)) == 0, "train и test пересекаются"
    assert len(X_train) + len(X_test) == len(X)

    # stratify сохранил долю ушедших почти одинаковой
    assert abs(y_train.mean() - y_test.mean()) < 0.005

    # Метрики пересчитываем вручную из матрицы ошибок
    for name, r in metrics.iterrows():
        n = r["tn"] + r["fp"] + r["fn"] + r["tp"]
        assert n == len(y_test)
        assert np.isclose(r["accuracy"], (r["tp"] + r["tn"]) / n)
        if r["tp"] + r["fp"] > 0:
            assert np.isclose(r["precision"], r["tp"] / (r["tp"] + r["fp"]))
        assert np.isclose(r["recall"], r["tp"] / (r["tp"] + r["fn"]))

    # Dummy всегда говорит «остался», поэтому его accuracy = доля оставшихся в test
    assert np.isclose(metrics.loc["Dummy (most_frequent)", "accuracy"], 1 - y_test.mean())
    print("Все проверки пройдены.")


def main():
    for d in (IMG_DIR, RES_DIR):
        d.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    eda = explore(raw)
    sql_results = run_sql(raw)

    section("3. ПОДГОТОВКА ДАННЫХ")
    X, y = clean(raw)
    categorical = [c for c in X.columns if c not in NUMERIC]
    # stratify=y — в train и test будет одинаковая доля ушедших (~26.5%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Признаков: {X.shape[1]} (числовых {len(NUMERIC)}, категориальных {len(categorical)})")
    print(f"Train: {len(X_train)} строк, доля ушедших {y_train.mean():.1%}")
    print(f"Test:  {len(X_test)} строк, доля ушедших {y_test.mean():.1%}")

    models, metrics, probas = train_and_evaluate(X_train, X_test, y_train, y_test, categorical)
    anna_probs = example_customer(models, X.columns)
    thresholds = threshold_demo(y_test, probas["Логистическая регрессия"])
    coefs, imp = interpret(models, X_train, X_test, y_test)

    plot_contract(sql_results)
    plot_roc(y_test, probas, metrics)
    plot_importance(imp)
    print(f"\nГрафики сохранены в {IMG_DIR.name}/")

    self_check(raw, X, y, X_train, X_test, y_train, y_test, sql_results, metrics)

    summary = {
        "eda": eda,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics.round(4).reset_index().to_dict(orient="records"),
        "logreg_thresholds": thresholds.round(4).reset_index().to_dict(orient="records"),
        "example_customer_churn_proba": {k: round(v, 4) for k, v in anna_probs.items()},
    }
    (RES_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
