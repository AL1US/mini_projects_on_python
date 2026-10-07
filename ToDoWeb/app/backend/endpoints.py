from flask import Blueprint, render_template, request

from app.backend.models import Tasks
from app.backend.services import create_task
from utils.path import DATA_PATH

crud_bp = Blueprint("crud", __name__)

@crud_bp.post("/create_task")
def create_task_api():
    try:
        title = str(request.form.get("title"))

        if not title:
            return {"error": "Title is required"}, 400

        description = str(request.form.get("description"))

        task = Tasks(title=title, description=description)
        create_task(task=task, file_path=DATA_PATH)

        return {"status": "success", "message": "Task created"}, 201

    except Exception as e:
        return {"Error": str({e})}, 500