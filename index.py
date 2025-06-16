import os
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
    if 'archivos' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    files = request.files.getlist('archivos')
    results = []
    file_info = []
    #file_path="/app/"
    file_path="C:/Users/jsilvarojas/source/repos/JIND-Multi/"
    file_json= "config.json"
    for file in files:
        if file.filename == '':
            flash('No selected file')
            continue
        try:
            # if file.filename.endswith('.json'):
            #     print(file.filename)
            #     file_info.append({'filename': file.filename})
                # config = json.load(file)
                # mn.main()
                # result_main = run_main(data)
                # result_compare = compare_methods(data)
                # results.append(f"Main Result: {result_main}, Compare Result: {result_compare}")
            if file.filename.endswith('.h5ad'):
                print(file.filename)
                file_info.append({'filename': file.filename})
                file_path = file_path + file.filename
                print(file_path)
                file.save(file_path)

                with open(file_json, 'r') as file:
                    config = json.load(file)
                # data = ad.read_h5ad(file)
                config['PATH'] = file_path
                with open(file_json, 'w') as file:
                    json.dump(config, file, indent=4)

        except json.JSONDecodeError:
            flash(f"Invalid JSON in file: {file.filename}")
            continue
        except Exception as e:
            flash(f"Error processing file {file.filename}: {str(e)}")
            continue
        # mn.main(file_json)
        mn.main.run_with_args(file_json)
    return render_template('index.html', file_info=file_info)
    
if __name__ == '__main__':
    app.run(debug=True, port =5003)