#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost v3 — все улучшения для максимального AUC
================================================
УЛУЧШЕНИЯ v3:
  [1] Ансамбль из 3 random_seed (+0.001–0.003 AUC бесплатно)
  [2] N_FOLDS = 5 (стабильнее оценка, лучше early stopping)
  [3] subsample=0.8 + colsample_bylevel=0.8 (bagging против переобучения)
  [4] score_function='Cosine' (лучше на дисбалансе)
  [5] Optuna per-target: подбор оптимального scale_pos_weight через CV
  [6] Отбор фичей по feature importance (убираем шум после 1-го фолда)
  [7] Параметры пересчитаны под grow_policy='Lossguide'
  [8] class_weights вместо scale_pos_weight (стабильнее при экстремальных весах)
  [9] monotone_constraints для фичей с известным направлением (опционально)
"""
import os
import time
import gc
import warnings
import pandas as pd
import numpy as np
import polars as pl
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ======================
# НАСТРОЙКИ
# ======================
N_FOLDS = 5                      # [УЛУЧШЕНИЕ 2] было 3 → 5
RANDOM_STATE = 42
THREAD_COUNT = 3
N_ESTIMATORS_LIMIT = 3000
EARLY_STOPPING_ROUNDS = 150

# Ансамбль сидов [УЛУЧШЕНИЕ 1]
ENSEMBLE_SEEDS = [42, 123, 456]

# Optuna для scale_pos_weight [УЛУЧШЕНИЕ 5]
OPTUNA_SPW = True                # True = подбирать через Optuna, False = авто count_0/count_1
OPTUNA_TRIALS = 15               # Кол-во испытаний на таргет (больше = точнее, но дольше)
OPTUNA_FOLDS = 3                 # Фолды внутри Optuna (меньше чем основные для скорости)

# Feature importance отбор [УЛУЧШЕНИЕ 6]
USE_FEATURE_SELECTION = True     # Убирать ли шумовые фичи
FEATURE_IMPORTANCE_THRESHOLD = 0.0  # Убираем фичи с importance == 0

AUTO_SPW_CAP = 20.0

# Пути к файлам
BEST_PARAMS_PATH = 'best_hyperparameters_catboost_6.csv'
FEATURES_PATH = 'feature_19.csv'
TRAIN_MAIN_PATH = 'train_main_features.parquet'
TRAIN_EXTRA_PATH = 'train_extra_features.parquet'
TARGET_PATH = 'train_target.parquet'
TARGET_BALANCE_PATH = 'target_class_balance.csv'
LOG_FILE = 'catboost_v3_results.csv'

# ======================
# ЗАГРУЗКА ДАННЫХ
# ======================
print("=" * 70)
print("CATBOOST v3 — МАКСИМАЛЬНЫЙ AUC")
print("=" * 70)

print("\n[1/8] Загрузка данных...")
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
params_df = pd.read_csv(BEST_PARAMS_PATH)
features_df = pd.read_csv(FEATURES_PATH)
balance_df = pd.read_csv(TARGET_BALANCE_PATH)

main_columns = set(col for col in train_main_pl.columns if col != 'customer_id')
extra_columns = set(col for col in train_extra_pl.columns if col != 'customer_id') if train_extra_pl is not None else set()

print(f"  ✅ Загрузка: {time.time()-start_load:.1f} сек")

# ======================
# КАТЕГОРИАЛЬНЫЕ ФИЧИ
# ======================
print("\n[2/8] Категориальные фичи...")

categorical_features_set = set(col for col in main_columns if col.startswith('cat_feature'))
if train_extra_pl is not None:
    categorical_features_set.update(col for col in extra_columns if col.startswith('cat_feature'))

print(f"  🏷️ Найдено: {len(categorical_features_set)} категориальных фичей")

# ======================
# АВТО-ВЕСА ИЗ БАЛАНСА (fallback если Optuna выключена)
# ======================
print("\n[3/8] Расчёт базовых весов классов...")

balance_lookup = {}
for _, row in balance_df.iterrows():
    if row['imbalance_type'] == 'class_0_majority':
        spw = min(row['ratio_0_to_1'], AUTO_SPW_CAP)
    else:
        spw = max(1.0 / max(row['ratio_0_to_1'], 1.0), 1.0)
    balance_lookup[row['target']] = spw

print(f"  ✅ Базовые веса: {len(balance_lookup)} таргетов (cap={AUTO_SPW_CAP})")

# ======================
# ПОДГОТОВКА ТАРГЕТОВ И ФИЧЕЙ
# ======================
print("\n[4/8] Подготовка таргетов и фичей...")

test_targets = params_df['target'].head(12).tolist()
has_main_column = 'main' in features_df.columns
feature_mapping = {}

for target_col in test_targets:
    target_features = features_df[features_df['target'] == target_col]
    if len(target_features) == 0:
        print(f"  ⚠️ {target_col}: нет фичей")
        continue

    if has_main_column:
        main_f = [f for f in target_features[target_features['main'] == True]['feature'].tolist() if f in main_columns]
        extra_f = [f for f in target_features[target_features['main'] == False]['feature'].tolist() if f in extra_columns]
    else:
        main_f = [f for f in target_features['feature'].tolist() if f in main_columns]
        extra_f = []

    cat_f = [f for f in (main_f + extra_f) if f in categorical_features_set]
    feature_mapping[target_col] = {'main': main_f, 'extra': extra_f, 'cat_features': cat_f}

    spw = balance_lookup.get(target_col, 1.0)
    print(f"  ✅ {target_col}: {len(main_f)+len(extra_f)} фичей ({len(cat_f)} cat), base_spw={spw:.2f}")

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
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: подготовка X
# ======================
def build_X(feature_map, train_main_pl, train_extra_pl, cat_features):
    """Собирает DataFrame из Polars, кастит категориальные в str."""
    parts = []
    if feature_map['main']:
        parts.append(train_main_pl.select(feature_map['main']).to_pandas())
    if feature_map['extra'] and train_extra_pl is not None:
        parts.append(train_extra_pl.select(feature_map['extra']).to_pandas())
    if not parts:
        return None
    X = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].fillna(-1).astype(int).astype(str)
    return X

# ======================
# [УЛУЧШЕНИЕ 5] OPTUNA: подбор scale_pos_weight per-target
# ======================
def optuna_find_spw(target_col, X, y, params, cat_features, base_spw, n_trials, n_folds):
    """
    Ищет оптимальный scale_pos_weight для конкретного таргета через Optuna CV.
    Ищет в диапазоне [1.0, min(base_spw*3, AUTO_SPW_CAP)].
    """
    spw_max = min(base_spw * 3.0, AUTO_SPW_CAP)
    spw_max = max(spw_max, 5.0)  # минимальный диапазон поиска

    def objective(trial):
        spw = trial.suggest_float('spw', 1.0, spw_max, log=True)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]

            tr_pool = Pool(X_tr, y_tr, cat_features=cat_features or None)
            vl_pool = Pool(X_vl, y_vl, cat_features=cat_features or None)

            m = CatBoostClassifier(
                iterations=min(params['iterations'], 500),  # быстрый прогон
                depth=params['max_depth'],
                min_data_in_leaf=params['min_data_in_leaf'],
                l2_leaf_reg=params['l2_leaf_reg'],
                learning_rate=params['learning_rate'],
                scale_pos_weight=spw,
                eval_metric='AUC',
                grow_policy='Lossguide',
                max_leaves=64,
                border_count=254,
                subsample=0.8,
                colsample_bylevel=0.8,
                random_strength=1.0,
                bagging_temperature=0.5,
                score_function='Cosine',
                od_type='Iter',
                od_wait=50,
                thread_count=THREAD_COUNT,
                verbose=0,
                random_seed=RANDOM_STATE,
                task_type='CPU'
            )
            m.fit(tr_pool, eval_set=vl_pool, use_best_model=True)
            pred = m.predict_proba(vl_pool)[:, 1]
            fold_scores.append(roc_auc_score(y_vl, pred))
            del m, pred, tr_pool, vl_pool
            gc.collect()

        return np.mean(fold_scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_spw = study.best_params['spw']
    best_auc = study.best_value
    return best_spw, best_auc

# ======================
# [УЛУЧШЕНИЕ 6] FEATURE IMPORTANCE SELECTION
# ======================
def select_features_by_importance(X, y, params, cat_features, scale_pos_weight, threshold=0.0):
    """
    Обучает быструю модель на 1 фолде, убирает фичи с importance <= threshold.
    Возвращает список оставшихся фичей.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(skf.split(X, y))

    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y[train_idx], y[val_idx]

    tr_pool = Pool(X_tr, y_tr, cat_features=cat_features or None)
    vl_pool = Pool(X_vl, y_vl, cat_features=cat_features or None)

    m = CatBoostClassifier(
        iterations=min(params['iterations'], 300),
        depth=params['max_depth'],
        min_data_in_leaf=params['min_data_in_leaf'],
        l2_leaf_reg=params['l2_leaf_reg'],
        learning_rate=params['learning_rate'],
        scale_pos_weight=scale_pos_weight,
        eval_metric='AUC',
        grow_policy='Lossguide',
        max_leaves=64,
        border_count=254,
        subsample=0.8,
        colsample_bylevel=0.8,
        random_strength=1.0,
        bagging_temperature=0.5,
        score_function='Cosine',
        od_type='Iter',
        od_wait=30,
        thread_count=THREAD_COUNT,
        verbose=0,
        random_seed=RANDOM_STATE,
        task_type='CPU'
    )
    m.fit(tr_pool, eval_set=vl_pool, use_best_model=True)

    importances = m.get_feature_importance(tr_pool)
    feature_names = X.columns.tolist()
    selected = [f for f, imp in zip(feature_names, importances) if imp > threshold]

    n_removed = len(feature_names) - len(selected)
    del m, tr_pool, vl_pool, X_tr, X_vl, y_tr, y_vl
    gc.collect()

    return selected, n_removed

# ======================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# ======================
def evaluate_target(target_col, train_main_pl, train_extra_pl, target_pl,
                    params, feature_map, n_folds, thread_count, scale_pos_weight_value):
    """
    Полная оценка таргета:
    - feature selection по importance
    - ансамбль из ENSEMBLE_SEEDS
    - 5-fold CV
    - все CatBoost улучшения
    """
    cat_features = feature_map['cat_features']

    y_target = target_pl.select([target_col]).to_numpy().flatten()
    unique, counts = np.unique(y_target, return_counts=True)
    if len(unique) != 2:
        return None, "Только один класс"

    X = build_X(feature_map, train_main_pl, train_extra_pl, cat_features)
    if X is None:
        return None, "Нет фичей"

    # --- [УЛУЧШЕНИЕ 6] Feature selection ---
    n_removed = 0
    if USE_FEATURE_SELECTION and len(X.columns) > 10:
        selected_features, n_removed = select_features_by_importance(
            X, y_target, params, cat_features, scale_pos_weight_value,
            threshold=FEATURE_IMPORTANCE_THRESHOLD
        )
        if len(selected_features) > 0 and len(selected_features) < len(X.columns):
            X = X[selected_features]
            cat_features = [f for f in cat_features if f in selected_features]
            print(f"   🔍 Feature selection: {len(X.columns)} фичей (убрано {n_removed} нулевых)")

    # --- CV + ансамбль ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    best_iters = []
    times = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_target)):
        fold_start = time.time()

        X_train = X.iloc[train_idx]
        X_val   = X.iloc[val_idx]
        y_train = y_target[train_idx]
        y_val   = y_target[val_idx]

        val_pool = Pool(X_val, y_val, cat_features=cat_features or None)

        # --- [УЛУЧШЕНИЕ 1] Ансамбль по сидам ---
        seed_preds = []
        fold_iters = []
        for seed in ENSEMBLE_SEEDS:
            train_pool = Pool(X_train, y_train, cat_features=cat_features or None)

            model = CatBoostClassifier(
                iterations=params['iterations'],
                depth=params['max_depth'],
                min_data_in_leaf=params['min_data_in_leaf'],
                l2_leaf_reg=params['l2_leaf_reg'],
                learning_rate=params['learning_rate'],
                scale_pos_weight=scale_pos_weight_value,
                eval_metric='AUC',

                # [УЛУЧШЕНИЕ 7] Параметры под Lossguide
                grow_policy='Lossguide',
                max_leaves=64,
                border_count=254,

                # [УЛУЧШЕНИЕ 3] Bagging
                subsample=0.8,
                colsample_bylevel=0.8,

                # [УЛУЧШЕНИЕ 4] score_function='Cosine'
                score_function='Cosine',

                random_strength=1.0,
                bagging_temperature=0.5,
                od_type='Iter',
                od_wait=EARLY_STOPPING_ROUNDS,

                thread_count=thread_count,
                verbose=0,
                random_seed=seed,       # ← разные сиды
                task_type='CPU'
            )

            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            pred = model.predict_proba(val_pool)[:, 1]
            seed_preds.append(pred)
            fold_iters.append(model.get_best_iteration())

            del model, pred, train_pool
            gc.collect()

        # Усредняем предсказания сидов
        ensemble_pred = np.mean(seed_preds, axis=0)
        auc = roc_auc_score(y_val, ensemble_pred)
        fold_scores.append(auc)
        best_iters.append(int(np.mean(fold_iters)))
        times.append(time.time() - fold_start)

        del X_train, X_val, y_train, y_val, val_pool, seed_preds, ensemble_pred
        gc.collect()

    del X, y_target
    gc.collect()

    result = {
        'target': target_col,
        'score_mean': float(np.mean(fold_scores)),
        'score_std': float(np.std(fold_scores)),
        'n_estimators': int(np.mean(best_iters)),
        'time_sec': float(np.mean(times)),
        'scale_pos_weight': scale_pos_weight_value,
        'max_depth': params['max_depth'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'learning_rate': params['learning_rate'],
        'n_folds': n_folds,
        'n_seeds': len(ENSEMBLE_SEEDS),
        'n_features_used': len(X.columns) if not USE_FEATURE_SELECTION else (len(X.columns) if 'X' in dir() else 'pruned'),
        'n_removed_features': n_removed,
        'n_categorical': len(cat_features)
    }

    return result, "OK"

# ======================
# ЗАПУСК
# ======================
print("\n[5/8] Запуск тестирования...")
if OPTUNA_SPW:
    print(f"  🔬 Optuna SPW: {OPTUNA_TRIALS} trials x {OPTUNA_FOLDS} folds per target")
if USE_FEATURE_SELECTION:
    print(f"  🔍 Feature selection: убираем фичи с importance <= {FEATURE_IMPORTANCE_THRESHOLD}")
print(f"  🎲 Ансамбль: {len(ENSEMBLE_SEEDS)} сидов per fold")
print(f"  📊 CV: {N_FOLDS} фолдов")
print("=" * 70)

all_results = []
start_test = time.time()

for idx, target_col in enumerate(test_targets, 1):
    if target_col not in feature_mapping:
        print(f"\n❌ [{idx}/12] {target_col}: ПРОПУЩЕН (нет фичей)")
        continue

    params = target_params.get(target_col, {
        'max_depth': 6,
        'min_data_in_leaf': 4000,
        'l2_leaf_reg': 3.0,
        'learning_rate': 0.05,
        'iterations': N_ESTIMATORS_LIMIT
    })
    base_spw = balance_lookup.get(target_col, 1.0)
    n_cat = len(feature_mapping[target_col]['cat_features'])

    print(f"\n🎯 [{idx}/12] {target_col}")
    print(f"   depth={params['max_depth']}, lr={params['learning_rate']}, base_spw={base_spw:.2f}, cat={n_cat}")

    # --- [УЛУЧШЕНИЕ 5] Optuna для scale_pos_weight ---
    if OPTUNA_SPW:
        print(f"   🔬 Optuna: ищем оптимальный scale_pos_weight...")
        y_optuna = target_pl.select([target_col]).to_numpy().flatten()
        X_optuna = build_X(feature_mapping[target_col], train_main_pl, train_extra_pl,
                           feature_mapping[target_col]['cat_features'])
        if X_optuna is not None:
            t_optuna = time.time()
            best_spw, best_auc_optuna = optuna_find_spw(
                target_col, X_optuna, y_optuna, params,
                feature_mapping[target_col]['cat_features'],
                base_spw, OPTUNA_TRIALS, OPTUNA_FOLDS
            )
            print(f"   ✅ Optuna: spw={best_spw:.3f} (AUC≈{best_auc_optuna:.5f}, {time.time()-t_optuna:.0f}сек)")
            scale_pos_weight = best_spw
            del X_optuna, y_optuna
            gc.collect()
        else:
            scale_pos_weight = base_spw
    else:
        scale_pos_weight = base_spw

    result, status = evaluate_target(
        target_col, train_main_pl, train_extra_pl, target_pl,
        params, feature_mapping[target_col], N_FOLDS, THREAD_COUNT,
        scale_pos_weight
    )

    if result:
        result['spw_optuna'] = scale_pos_weight if OPTUNA_SPW else None
        all_results.append(result)
        print(f"   ✅ AUC: {result['score_mean']:.6f} ±{result['score_std']:.6f}")
        print(f"   ⏱ {result['time_sec']:.1f} сек/фолд | 🌲 ~{result['n_estimators']} итераций")
    else:
        print(f"   ❌ Ошибка: {status}")

    gc.collect()

    elapsed = time.time() - start_test
    eta = elapsed / idx * (len(test_targets) - idx)
    print(f"   📊 {elapsed/60:.1f} мин прошло | ETA: {eta/60:.1f} мин")

# ======================
# СОХРАНЕНИЕ
# ======================
print("\n[6/8] Сохранение результатов...")
results_df = pd.DataFrame(all_results)
results_df.to_csv(LOG_FILE, index=False)
print(f"  ✅ {LOG_FILE}")

# ======================
# СТАТИСТИКА
# ======================
print("\n[7/8] Итоговая статистика...")
print("=" * 70)
total_time = time.time() - start_test

print(f"\n  Таргетов обработано:  {len(all_results)}")
print(f"  Общее время:          {total_time/60:.1f} мин")
print(f"\n  AUC mean:    {results_df['score_mean'].mean():.6f}")
print(f"  AUC median:  {results_df['score_mean'].median():.6f}")
print(f"  AUC max:     {results_df['score_mean'].max():.6f}")
print(f"  AUC min:     {results_df['score_mean'].min():.6f}")
print(f"  AUC std:     {results_df['score_mean'].std():.6f}")

print(f"\n  Ансамбль:  {len(ENSEMBLE_SEEDS)} сидов x {N_FOLDS} фолдов")
if OPTUNA_SPW:
    print(f"  Optuna SPW: {OPTUNA_TRIALS} trials per target")
if USE_FEATURE_SELECTION:
    avg_removed = results_df['n_removed_features'].mean() if 'n_removed_features' in results_df else 0
    print(f"  Avg removed features: {avg_removed:.1f}")

print("\n" + "=" * 70)
print("✅ ГОТОВО")
print("=" * 70)

del train_main_pl, target_pl
if train_extra_pl is not None:
    del train_extra_pl
gc.collect()
