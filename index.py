import os
import json
import secrets
import jsonify
import anndata as ad
import subprocess
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
            if file.filename.endswith('.h5ad'):
                print(file.filename)
                file_info.append({'filename': file.filename})
                file_path = os.path.join(file_path, file.filename)
                print(file_path)
                file.save(file_path)

                # Check if the JSON file exists and is not empty
                if os.path.getsize(file_json) == 0:
                    flash(f"The JSON configuration file {file_json} is empty.")
                    continue

                with open(file_json, 'r') as json_file:
                    try:
                        config = json.load(json_file)
                    except json.JSONDecodeError as e:
                        flash(f"Invalid JSON in file: {file_json} - {str(e)}")
                        continue

                config['PATH'] = file_path
                with open(file_json, 'w') as json_file:
                    json.dump(config, json_file, indent=4)

                command = ['run-jind-multi', '--config', file_json]
                print(command)
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                print(result)
                results.append(f"Command executed successfully: {result.stdout}")

        except subprocess.CalledProcessError as e:
            error_message = (
                f"Error executing command: {e}\n"
                f"Return code: {e.returncode}\n"
                f"Command: {e.cmd}\n"
                f"Output: {e.output}\n"
                f"Error output: {e.stderr}\n"
            )
            flash(error_message)
            continue
        except Exception as e:
            flash(f"Error processing file {file.filename}: {str(e)}")
            results.append(f"Unexpected error: {str(e)}")
            continue

    return render_template('index.html', file_info=file_info, results=results)


if __name__ == '__main__':
    app.run(debug=True, port =5003)