import json
import secrets
import anndata as ad
from flask import  Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap5
import jind_multi.main as mn


app = Flask(__name__)
secret_key = secrets.token_hex(32)
print("Your secret key:", secret_key)
app.secret_key = secret_key
bootstrap = Bootstrap5(app)

@app.route("/", methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])

def uploadf():
    print("entro")
    if 'archivos' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    files = request.files.getlist('archivos')
    results = []
    file_info = []

    for file in files:
        if file.filename == '':
            flash('No selected file')
            continue
        try:
            if file.filename.endswith('.json'):
                print(file.filename)
                file_info.append({'filename': file.filename})
                config = json.load(file)
                # mn.main()
                # result_main = run_main(data)
                # result_compare = compare_methods(data)
                # results.append(f"Main Result: {result_main}, Compare Result: {result_compare}")
            elif file.filename.endswith('.h5ad'):
                print(file.filename)
                file_info.append({'filename': file.filename})
                data = ad.read_h5ad(file)

        except json.JSONDecodeError:
            flash(f"Invalid JSON in file: {file.filename}")
            continue
        except Exception as e:
            flash(f"Error processing file {file.filename}: {str(e)}")
            continue

    return render_template('index.html', file_info=file_info)
    
if __name__ == '__main__':
    app.run(debug=True, port =5003)