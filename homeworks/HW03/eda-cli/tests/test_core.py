from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(df, summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0

    # Проверяем наличие новых флагов
    new_flags = [
        "has_constant_columns",
        "has_high_cardinality_categoricals", 
        "has_suspicious_id_duplicates",
        "has_many_zero_values",
    ]
    
    for flag in new_flags:
        assert flag in flags, f"Отсутствует флаг: {flag}"


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

#Новое для добавленных эвристик

def test_has_constant_columns():
    """Тест для выявления константных колонок."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [5, 5, 5, 5],  # все значения одинаковые
        "variable_col": [1, 2, 3, 4],
    })
    summary = summarize_dataset(df)
    missing = missing_table(df)
    
    try:
        flags = compute_quality_flags(df, summary, missing)
    except TypeError:
        flags = compute_quality_flags(summary, missing)
    # Проверяем, что флаг сработал
    assert flags["has_constant_columns"] == True, \
        f"Ожидалось True для константной колонки, получено {flags['has_constant_columns']}"
    
    # Проверяем датасет без константных колонок
    df_no_constant = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    })
    summary2 = summarize_dataset(df_no_constant)
    missing2 = missing_table(df_no_constant)
    
    try:
        flags2 = compute_quality_flags(df_no_constant, summary2, missing2)
    except TypeError:
        flags2 = compute_quality_flags(summary2, missing2)
    
    assert flags2["has_constant_columns"] == False, \
        "В датасете без константных колонок флаг должен быть False"


def test_has_high_cardinality_categoricals():
    """Тест для выявления высокой кардинальности категориальных признаков."""
    # Датсет с высокой кардинальностью
    df_high = pd.DataFrame({
        "id": range(50),
        "high_card": [f"val_{i}" for i in range(50)],  # все уникальные
        "city": ["Moscow", "SPb"] * 25,  # низкая кардинальность
    })
    summary = summarize_dataset(df_high)
    missing = missing_table(df_high)
    
    try:
        flags = compute_quality_flags(df_high, summary, missing)
    except TypeError:
        flags = compute_quality_flags(summary, missing)
    
    # Флаг должен быть True
    assert flags["has_high_cardinality_categoricals"] == True, \
        f"Ожидалось True для высокой кардинальности, получено {flags['has_high_cardinality_categoricals']}"
    
    # Датсет с нормальной кардинальностью
    df_low = pd.DataFrame({
        "city": ["Moscow", "SPb", "Moscow", "Kazan"],
        "status": ["active", "inactive", "active", "pending"],
    })
    summary2 = summarize_dataset(df_low)
    missing2 = missing_table(df_low)
    
    try:
        flags2 = compute_quality_flags(df_low, summary2, missing2)
    except TypeError:
        flags2 = compute_quality_flags(summary2, missing2)
    
    assert flags2["has_high_cardinality_categoricals"] == False, \
        "В датасете с нормальной кардинальностью флаг должен быть False"


def test_has_suspicious_id_duplicates():
    """Тест для выявления дубликатов ID."""
    # Датсет с дубликатами ID
    df_duplicates = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2, 4],  # дубликаты 1 и 2
        "value": [10, 20, 30, 40, 50, 60],
    })
    summary = summarize_dataset(df_duplicates)
    missing = missing_table(df_duplicates)
    
    try:
        flags = compute_quality_flags(df_duplicates, summary, missing)
    except TypeError:
        flags = compute_quality_flags(summary, missing)
    
    assert flags["has_suspicious_id_duplicates"] == True, \
        f"Ожидалось True для дубликатов ID, получено {flags['has_suspicious_id_duplicates']}"
    
    # Датсет без дубликатов
    df_unique = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50],
    })
    
    summary2 = summarize_dataset(df_unique)
    missing2 = missing_table(df_unique)
    
    try:
        flags2 = compute_quality_flags(df_unique, summary2, missing2)
    except TypeError:
        flags2 = compute_quality_flags(summary2, missing2)
    
    assert flags2["has_suspicious_id_duplicates"] == False, \
        "В датасете без дубликатов флаг должен быть False"


def test_has_many_zero_values():
    """Тест для выявления колонок с большим количеством нулей."""
    # Датсет с большим количеством нулей
    df_many_zeros = pd.DataFrame({
        "id": range(10),
        "mostly_zeros": [0, 0, 0, 0, 0, 0, 0, 1, 2, 3],  # 70% нулей
        "few_zeros": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],     # 10% нулей
    })
    
    summary = summarize_dataset(df_many_zeros)
    missing = missing_table(df_many_zeros)
    
    try:
        flags = compute_quality_flags(df_many_zeros, summary, missing)
    except TypeError:
        flags = compute_quality_flags(summary, missing)
    
    assert flags["has_many_zero_values"] == True, \
        f"Ожидалось True для многих нулей, получено {flags['has_many_zero_values']}"
    
    # Датсет с нормальным количеством нулей
    df_few_zeros = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [0, 1, 2, 3, 0],  # 40% нулей (но порог 50%)
    })
    
    summary2 = summarize_dataset(df_few_zeros)
    missing2 = missing_table(df_few_zeros)
    
    try:
        flags2 = compute_quality_flags(df_few_zeros, summary2, missing2)
    except TypeError:
        flags2 = compute_quality_flags(summary2, missing2)
    
    # Проверяем что флаг False (т.к. порог 50%)
    assert flags2["has_many_zero_values"] == False, \
        "В датасете с <70% нулей флаг должен быть False"


def test_quality_score_with_new_heuristics():
    """Тест для проверки влияния новых эвристик на quality_score."""
    # Создаём датасет с несколькими проблемами
    df_problems = pd.DataFrame({
        "id": [1, 2, 1, 2],           # дубликаты ID
        "constant": [7, 7, 7, 7],     # константная колонка
        "high_card": ["a", "b", "c", "d"],  # все уникальные
        "many_zeros": [0, 0, 0, 1],   # 75% нулей
    })
    
    summary = summarize_dataset(df_problems)
    missing = missing_table(df_problems)
    
    try:
        flags = compute_quality_flags(df_problems, summary, missing)
    except TypeError:
        flags = compute_quality_flags(summary, missing)
    
    assert flags["has_constant_columns"] == True
    assert flags["has_high_cardinality_categoricals"] == True
    assert flags["has_suspicious_id_duplicates"] == True
    assert flags["has_many_zero_values"] == True
    
    # Quality score должен быть низким
    assert "quality_score" in flags
    score = flags["quality_score"]
    
    # Проверяем границы
    assert 0 <= score <= 100, f"quality_score должен быть 0-100, получено {score}"
    
    # Проверяем датасет без проблем
    df_clean = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [10.5, 20.3, 15.7, 18.2],
        "category": ["A", "B", "A", "B"],
    })
    
    summary2 = summarize_dataset(df_clean)
    missing2 = missing_table(df_clean)
    
    try:
        flags2 = compute_quality_flags(df_clean, summary2, missing2)
    except TypeError:
        flags2 = compute_quality_flags(summary2, missing2)
    
    assert flags2["has_constant_columns"] == False
    assert flags2["has_high_cardinality_categoricals"] == False
    assert flags2["has_suspicious_id_duplicates"] == False
    assert flags2["has_many_zero_values"] == False
    
    # Quality score должен быть выше
    if "quality_score" in flags2:
        score2 = flags2["quality_score"]
        assert score2 > score, \
            f"Чистый датасет должен иметь более высокий score: {score2} > {score}"


def test_function_signatures():
    """Тест для проверки разных сигнатур функции compute_quality_flags."""
    df = _sample_df()
    summary = summarize_dataset(df)
    missing = missing_table(df)
    try:
        flags_new = compute_quality_flags(df, summary, missing)
        assert isinstance(flags_new, dict)
    except TypeError:
        pass  # Если функция ожидает 2 аргумента
