#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестирование 12 таргетов с eval_metric='AUC' и scale_pos_weight
ИСПРАВЛЕНО:
  1. cat_features передаются как строки через Pool, а не индексы через numpy
  2. scale_pos_weight = авто-рассчитывается для каждого таргета (не фиксированный 10)
  3. Убран cast в Int32 — CatBoost сам обрабатывает строковые/int категории
  4. Добавлены border_count, random_strength, bagging_temperature для CatBoost-специфичных улучшений
  5. grow_policy=Lossguide (аналог LightGBM leaf-wise) вместо дефолтного SymmetricTree
  6. Исправлена передача фичей через Pool (DataFrame с именами, не numpy array)
"""
import os
import time
import gc
import warnings
import pandas as pd
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

# ======================
# НАСТРОЙКИ
# ======================
N_FOLDS = 3
RANDOM_STATE = 42
THREAD_COUNT = 3
N_ESTIMATORS_LIMIT = 3000
EARLY_STOPPING_ROUNDS = 150      # ← Увеличено: CatBoost медленнее «разгоняется»

# ВАЖНО: scale_pos_weight теперь авто-рассчитывается для КАЖДОГО таргета
# Но для очень несбалансированных (топ-N) применяем cap
TOP_N_IMBALANCED_TARGETS = 10
AUTO_SPW_CAP = 20.0              # ← Максимальный авто-вес (защита от переобучения)

# Пути к файлам
BEST_PARAMS_PATH = 'best_hyperparameters_catboost_6.csv'
FEATURES_PATH = 'feature_19.csv'
TRAIN_MAIN_PATH = 'train_main_features.parquet'
TRAIN_EXTRA_PATH = 'train_extra_features.parquet'
TARGET_PATH = 'train_target.parquet'
TARGET_BALANCE_PATH = 'target_class_balance.csv'
LOG_FILE = 'test_12_targets_auc_log.csv'

# ======================
# ЗАГРУЗКА ДАННЫХ
# ======================
print("=" * 70)
print("ТЕСТ 12 ТАРГЕТОВ: CATBOOST ИСПРАВЛЕННЫЙ")
print("=" * 70)

print("\n[1/7] Загрузка данных...")
start_load = time.time()

train_main_pl = pl.read_parquet(TRAIN_MAIN_PATH)
print(f"  ✅ train_main: {train_main_pl.shape}")

if os.path.exists(TRAIN_EXTRA_PATH):
    train_extra_pl = pl.read_parquet(TRAIN_EXTRA_PATH)
    print(f"  ✅ train_extra: {train_extra_pl.shape}")
else:
    train_extra_pl = None
    print(f"  ⚠️ train_extra не найден")

target_pl = pl.read_parquet(TARGET_PATH)
print(f"  ✅ targets: {target_pl.shape}")

params_df = pd.read_csv(BEST_PARAMS_PATH)
features_df = pd.read_csv(FEATURES_PATH)
balance_df = pd.read_csv(TARGET_BALANCE_PATH)

main_columns = set(col for col in train_main_pl.columns if col != 'customer_id')
extra_columns = set(col for col in train_extra_pl.columns if col != 'customer_id') if train_extra_pl is not None else set()

# ======================
# ОПРЕДЕЛЕНИЕ КАТЕГОРИАЛЬНЫХ ФИЧЕЙ
# ИСПРАВЛЕНИЕ #1: НЕ кастим в Int32 — оставляем как есть.
# CatBoost принимает строки, int, float для cat_features через Pool.
# Numpy array теряет имена колонок → CatBoost не может найти cat по индексу правильно.
# ======================
print("\n[2/7] Определение категориальных фичей (без каста типов)...")

categorical_features_set = set(col for col in main_columns if col.startswith('cat_feature'))
if train_extra_pl is not None:
    extra_categorical = set(col for col in extra_columns if col.startswith('cat_feature'))
    categorical_features_set.update(extra_categorical)

print(f"  🏷️ Категориальных фичей всего: {len(categorical_features_set)}")
print(f"  ℹ️  Типы не изменяем — CatBoost обработает сам через Pool")

# ======================
# АВТО-РАСЧЁТ scale_pos_weight ДЛЯ КАЖДОГО ТАРГЕТА
# ИСПРАВЛЕНИЕ #2: Фиксированный вес=10 для всех несбалансированных — неверно.
# Правильно: count_0 / count_1 (или cap'd версия).
# ======================
print("\n[3/7] Расчёт авто-весов для каждого таргета...")

balance_lookup = {}
for _, row in balance_df.iterrows():
    if row['imbalance_type'] == 'class_0_majority':
        spw = min(row['ratio_0_to_1'], AUTO_SPW_CAP)
    else:
        # class_1_majority — инвертируем
        spw = 1.0 / max(row['ratio_0_to_1'], 1.0)
        spw = max(spw, 1.0)
    balance_lookup[row['target']] = spw

print(f"  ✅ scale_pos_weight рассчитан для {len(balance_lookup)} таргетов")
print(f"  ℹ️  Cap = {AUTO_SPW_CAP} (защита от переобучения на редком классе)")

# ======================
# ПОДГОТОВКА ТАРГЕТОВ
# ======================
print("\n[4/7] Подготовка 12 таргетов...")

test_targets = params_df['target'].head(12).tolist()
has_main_column = 'main' in features_df.columns
feature_mapping = {}

for target_col in test_targets:
    target_features = features_df[features_df['target'] == target_col]
    if len(target_features) == 0:
        print(f"  ⚠️ {target_col}: нет фичей")
        continue

    if has_main_column:
        main_features = target_features[target_features['main'] == True]['feature'].tolist()
        extra_features = target_features[target_features['main'] == False]['feature'].tolist()
        valid_main = [f for f in main_features if f in main_columns]
        valid_extra = [f for f in extra_features if f in extra_columns]
    else:
        valid_main = [f for f in target_features['feature'].tolist() if f in main_columns]
        valid_extra = []

    cat_in_features = [f for f in (valid_main + valid_extra) if f in categorical_features_set]

    feature_mapping[target_col] = {
        'main': valid_main,
        'extra': valid_extra,
        'cat_features': cat_in_features      # ← список имён, не индексов
    }

    spw = balance_lookup.get(target_col, 1.0)
    print(f"  ✅ {target_col}: {len(valid_main)+len(valid_extra)} фичей "
          f"({len(cat_in_features)} cat), spw={spw:.2f}")

# Параметры из файла
target_params = {}
for _, row in params_df.iterrows():
    if row['target'] in test_targets:
        target_params[row['target']] = {
            'max_depth': int(row['max_depth']),
            'min_data_in_leaf': int(row['min_data_in_leaf']),
            'l2_leaf_reg': float(row['l2_leaf_reg']),
            'learning_rate': float(row['learning_rate']),
            'iterations': int(row['iterations'])
        }

# ======================
# ФУНКЦИЯ ОЦЕНКИ
# ======================
def evaluate_target(target_col, train_main_pl, train_extra_pl, target_pl,
                    params, feature_map, n_folds, thread_count,
                    scale_pos_weight_value):
    """
    ИСПРАВЛЕНИЯ внутри функции:
    - Данные передаются через Pool с DataFrame (сохраняются имена колонок)
    - cat_features передаются как список имён строк
    - grow_policy='Lossguide' — leaf-wise как в LightGBM
    - border_count=254 — больше сплитов для числовых фичей
    - random_strength и bagging_temperature — регуляризация в стиле CatBoost
    """
    list_main = feature_map['main']
    list_extra = feature_map['extra']
    cat_features = feature_map['cat_features']   # ← имена строками

    if len(list_main) == 0 and len(list_extra) == 0:
        return None, "Нет фичей"

    y_target = target_pl.select([target_col]).to_numpy().flatten()

    # ИСПРАВЛЕНИЕ #3: Строим DataFrame с именами колонок, а не numpy array.
    # Это позволяет передавать cat_features как список имён (надёжнее индексов).
    parts = []
    if list_main:
        parts.append(train_main_pl.select(list_main).to_pandas())
    if list_extra and train_extra_pl is not None:
        parts.append(train_extra_pl.select(list_extra).to_pandas())

    if not parts:
        return None, "Нет фичей"

    X = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    # Для cat_features: CatBoost требует строки или int, не float
    # Заполняем NaN в категориальных колонках строкой 'nan'
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna(-1).astype(int).astype(str)

    unique, counts = np.unique(y_target, return_counts=True)
    if len(unique) != 2:
        return None, "Только один класс"

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    best_iters = []
    times = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_target)):
        fold_start = time.time()

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y_target[train_idx]
        y_val = y_target[val_idx]

        # ИСПРАВЛЕНИЕ #4: Pool с именами колонок + cat_features как строки
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features if cat_features else None
        )
        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=cat_features if cat_features else None
        )

        # ИСПРАВЛЕНИЕ #5: Добавлены CatBoost-специфичные параметры
        model = CatBoostClassifier(
            iterations=params['iterations'],
            depth=params['max_depth'],
            min_data_in_leaf=params['min_data_in_leaf'],
            l2_leaf_reg=params['l2_leaf_reg'],
            learning_rate=params['learning_rate'],
            scale_pos_weight=scale_pos_weight_value,
            eval_metric='AUC',

            # ↓ НОВЫЕ ПАРАМЕТРЫ ↓
            grow_policy='Lossguide',        # leaf-wise как LightGBM (вместо SymmetricTree)
            max_leaves=64,                  # аналог num_leaves в LightGBM
            border_count=254,               # больше сплитов для числовых фичей (default=128)
            random_strength=1.0,            # случайность при выборе сплита (регуляризация)
            bagging_temperature=0.5,        # bootstrap по Байесу (0=детерминировано, 1=стандарт)
            od_type='Iter',                 # early stopping по итерациям
            od_wait=EARLY_STOPPING_ROUNDS,
            # ↑ НОВЫЕ ПАРАМЕТРЫ ↑

            thread_count=thread_count,
            verbose=0,
            random_seed=RANDOM_STATE,
            task_type='CPU'
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True
        )

        pred = model.predict_proba(val_pool)[:, 1]
        auc = roc_auc_score(y_val, pred)
        scores.append(auc)
        best_iters.append(model.get_best_iteration())
        times.append(time.time() - fold_start)

        del model, pred, X_train, X_val, y_train, y_val, train_pool, val_pool
        gc.collect()

    del X, y_target
    gc.collect()

    result = {
        'target': target_col,
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'n_estimators': int(np.mean(best_iters)),
        'time_sec': np.mean(times),
        'scale_pos_weight': scale_pos_weight_value,
        'max_depth': params['max_depth'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'learning_rate': params['learning_rate'],
        'n_folds': n_folds,
        'n_categorical': len(cat_features)
    }

    return result, "OK"

# ======================
# ЗАПУСК
# ======================
print("\n[5/7] Запуск тестирования...")
print("=" * 70)

all_results = []
start_test = time.time()

for idx, target_col in enumerate(test_targets, 1):
    if target_col not in feature_mapping:
        print(f"\n❌ [{idx}/12] {target_col}: ПРОПУЩЕН (нет фичей)")
        continue

    scale_pos_weight = balance_lookup.get(target_col, 1.0)
    params = target_params.get(target_col, {
        'max_depth': 6,
        'min_data_in_leaf': 4000,
        'l2_leaf_reg': 3.0,
        'learning_rate': 0.05,
        'iterations': N_ESTIMATORS_LIMIT
    })

    n_cat = len(feature_mapping[target_col]['cat_features'])
    print(f"\n🎯 [{idx}/12] {target_col}")
    print(f"   depth={params['max_depth']}, min_data={params['min_data_in_leaf']}, "
          f"lr={params['learning_rate']}, spw={scale_pos_weight:.2f}, cat={n_cat}")

    result, status = evaluate_target(
        target_col, train_main_pl, train_extra_pl, target_pl,
        params, feature_mapping[target_col], N_FOLDS, THREAD_COUNT,
        scale_pos_weight
    )

    if result:
        all_results.append(result)
        print(f"   ✅ AUC: {result['score_mean']:.6f} ±{result['score_std']:.6f}")
        print(f"   ⏱ {result['time_sec']:.1f} сек/фолд | 🌲 {result['n_estimators']} итераций")
    else:
        print(f"   ❌ Ошибка: {status}")

    gc.collect()

    elapsed = time.time() - start_test
    eta = elapsed / idx * (len(test_targets) - idx)
    print(f"   📊 {elapsed/60:.1f} мин прошло, ETA: {eta/60:.1f} мин")

# ======================
# СОХРАНЕНИЕ И СТАТИСТИКА
# ======================
print("\n[6/7] Сохранение результатов...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(LOG_FILE, index=False)
print(f"  ✅ Сохранено в: {LOG_FILE}")

print("\n[7/7] Итоговая статистика...")
print("=" * 70)
total_time = time.time() - start_test
print(f"  Обработано таргетов: {len(all_results)}")
print(f"  Общее время: {total_time/60:.1f} мин")
print(f"\n  AUC mean:   {results_df['score_mean'].mean():.6f}")
print(f"  AUC median: {results_df['score_mean'].median():.6f}")
print(f"  AUC max:    {results_df['score_mean'].max():.6f}")
print(f"  AUC min:    {results_df['score_mean'].min():.6f}")

print("\n" + "=" * 70)
print("✅ ГОТОВО")
print("=" * 70)

del train_main_pl, target_pl, results_df
if train_extra_pl is not None:
    del train_extra_pl
gc.collect()
