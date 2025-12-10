# -*- coding: windows-1251 -*-`n`n
import pandas as pd
import numpy as np
import os
from core import compute_quality_flags, summarize_dataset, missing_table

def test_on_example_csv():
    project_path = r'C:\WINDOWS\system32\eaa-git-rep\eaa_digital_department_gr2\homeworks\HW03\eda-cli'
    csv_path = os.path.join(project_path, 'data', 'example.csv')
    
    df = pd.read_csv(csv_path)

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    flags = compute_quality_flags(df, summary, missing_df)
    
    print("Результаты для example.csv:")
    for key, value in flags.items():
        print(f"  {key}: {value}")
    
    return flags

def create_test_dataset():
    # Создаём DataFrame с преднамеренными проблемами
    test_data = {
        # ПРОБЛЕМА 1: Константная колонка
        'constant_col': [1] * 50,  # все значения = 1
        # ПРОБЛЕМА 2: Высокая кардинальность
        'high_cardinality_col': [f'user_{i}' for i in range(50)],  # все уникальные
        # ПРОБЛЕМА 3: Дубликаты ID
        'user_id': list(range(25)) + list(range(25)),  # 0-24 дважды
        # ПРОБЛЕМА 4: Много нулей
        'many_zeros': [0] * 40 + [1] * 10,  # 80% нулей
        # ПРОБЛЕМА 5: Пропуски
        'with_missing': [1, 2, np.nan, 4, 5] * 10,
        # Нормальные колонки
        'normal_cat': ['A', 'B', 'C'] * 16 + ['A', 'B'],  # 3 категории
        'normal_num': list(range(50)),
    }
    
    df_test = pd.DataFrame(test_data)
    
    summary_test = summarize_dataset(df_test)
    missing_df_test = missing_table(df_test)
    
    flags_test = compute_quality_flags(df_test, summary_test, missing_df_test)
    
    print("Результаты для тестового датасета:")
    for key, value in flags_test.items():
        print(f"  {key}: {value}")
    
    return df_test, flags_test

if __name__ == "__main__":
    flags1 = test_on_example_csv()
    df_test, flags2 = create_test_dataset()
    
    # Сохраняем тестовый датасет в CSV
    df_test.to_csv('test_1zad.csv', index=False)
    print("Тестовый датасет сохранён в test_1zad.csv")