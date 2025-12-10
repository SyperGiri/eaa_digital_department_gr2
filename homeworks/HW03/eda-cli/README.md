# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор датасета

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт с настраиваемыми параметрами

```bash
uv run eda-cli report data/example.csv --out-dir reports
```
с новыми параметрами

```bash
eda-cli report data/example.csv \
  --title "EDA-отчёт" \
  --max-hist-columns 9 \
  --top-k-categories 8 \
  --min-missing-share 0.1 \
  --out-dir my_report
```

Новые параметры report:
-- title - Заголовок отчёта в Markdown	(по умолчанию "EDA-отчёт")
--max-hist-columns - Максимальное количество числовых колонок для гистограмм	(по умолчанию 6)
--top-k-categories - Сколько top-значений выводить для категориальных признаков	(по умолчанию 5)
--min-missing-share - Порог доли пропусков для выделения проблемных колонок(по умолчанию	0.1 (10%))

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
