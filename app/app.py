from flask import Flask
from flask import request

app = Flask(__name__)

@app.route("/")
def index():
    return """
    <h1>Welcome to Perion AI Engine</h1>
    <p><a href='/api'>Access the API</a></p>
    """

@app.route("/api", methods=['GET'])
    def index():
        if request.method == 'GET':
            dataset_url = request.args.get('url', '')
            return dataset_url

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')