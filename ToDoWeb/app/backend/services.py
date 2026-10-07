import json
from app.backend.utils import get_data
from app.backend.models import Tasks
from pathlib import Path
import uuid

# Задачи
# Создание
def create_task(task: Tasks, file_path: Path):
    # Загрузить существующие данные
    data = get_data(file_path)

    # Перевод в json строку и добавляем объект
    data.append(task.model_dump(mode="json"))

    # Сохраняем обратно в файл
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Изменение


# Удаление

# Выполнение

