from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return """
    <h1>Test Docker</h1>
    <p>Works</p>
    """
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')