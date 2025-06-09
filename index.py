from flask import Flask, render_template, request 
from flask_bootstrap import Bootstrap5

app = Flask(__name__)
bootstrap = Bootstrap5(app)

@app.route("/", methods=['GET','POST'])
def home():
    if request.method == 'POST':
        f = request.files.get('archivo')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port =5003)