from flask import Flask, render_template

from ToDoWeb.utils.path import TEMPLATES_PATH

app = Flask(__name__, template_folder=TEMPLATES_PATH)

@app.route("/")
def index_page():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)