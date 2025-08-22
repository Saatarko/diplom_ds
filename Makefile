.PHONY: format isort clean_cashe black lint autoflake check

# Проверка только (без изменений)
check:
	isort . --check-only
	black . --check
	flake8 .
	nbqa isort . --check-only
	nbqa black . --check
	nbqa flake8

# Форматирование: isort → autoflake → black
format:
	isort .
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r .
	black .
	nbqa isort .
	nbqa black .
	nbqa flake8 .
# Только сортировка импортов
isort:
	isort .
	nbqa isort .

# Только black
black:
	black .
	nbqa black .

# Только удаление неиспользуемых импортов
autoflake:
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r .

# Проверка соответствия PEP8 и другим правилам
lint:
	flake8 .
	nbqa flake8 .

# Очистка временных и кэш-файлов
clean_cashe:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

