from flask import Flask, render_template, Blueprint
from utils.path import TEMPLATES_PATH, DATA_PATH
from app.backend.endpoints import crud_bp

app = Flask(__name__, template_folder=TEMPLATES_PATH)

app.register_blueprint(crud_bp)

@app.route("/")
def index_page():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)