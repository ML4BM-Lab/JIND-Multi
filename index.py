import os
import sys
import time
import json
import secrets
import subprocess
import threading
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_socketio import SocketIO,emit
from flask_bootstrap import Bootstrap5

app = Flask(__name__)
secret_key = secrets.token_hex(32)
print("Your secret key:", secret_key)
app.secret_key = secret_key
bootstrap = Bootstrap5(app)
socketio = SocketIO(app, cors_allowed_origins="*")
file_json = "/app/config.json"
# file_json = "config.json"

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route("/config", methods=['POST'])
def config():
    
    BATCH_COL =  request.form.get('campo1', '')
    LABELS_COL =  request.form.get('campo2', '')
    SOURCE_DATASET_NAME =  request.form.get('campo3', '')
    TARGET_DATASET_NAME =  request.form.get('campo4', '')
    INTER_DATASETS_NAMES =  request.form.get('campo5', '')
    EXCLUDE_DATASETS_NAMES =  request.form.get('campo6', '')
    MIN_CELL_TYPE_POPULATION =  request.form.get('campo7', '') 
    
    if BATCH_COL != '':
        config['BATCH_COL'] = BATCH_COL
        print(f"Valor de campo1: {BATCH_COL}")
    if LABELS_COL != '':
        config['LABELS_COL'] = LABELS_COL
        print(f"Valor de campo2: {LABELS_COL}")
    if SOURCE_DATASET_NAME != '':
        config['SOURCE_DATASET_NAME'] = SOURCE_DATASET_NAME
        print(f"Valor de campo3: {SOURCE_DATASET_NAME}")
    if BATCH_COL != '':
        config['TARGET_DATASET_NAME'] = TARGET_DATASET_NAME
        print(f"Valor de campo4: {TARGET_DATASET_NAME}")
    if BATCH_COL != '':
        config['INTER_DATASETS_NAMES'] = INTER_DATASETS_NAMES
        print(f"Valor de campo5: {INTER_DATASETS_NAMES}")
    if BATCH_COL != '':
        config['EXCLUDE_DATASETS_NAMES'] = EXCLUDE_DATASETS_NAMES
        print(f"Valor de campo6: {EXCLUDE_DATASETS_NAMES}")
    if BATCH_COL != '':
        config['MIN_CELL_TYPE_POPULATION'] = MIN_CELL_TYPE_POPULATION
        print(f"Valor de campo7: {MIN_CELL_TYPE_POPULATION}")

    if os.path.getsize(file_json) == 0:
        flash(f"The JSON configuration file {file_json} is empty.")
        

    with open(file_json, 'r') as json_file:
        try:
            config = json.load(json_file)
        except json.JSONDecodeError as e:
            flash(f"Invalid JSON in file: {file_json} - {str(e)}")
        
    with open(file_json, 'w') as json_file:
                json.dump(config, json_file, indent=4)

    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def uploadf():
    if 'archivos' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    files = request.files.getlist('archivos')
    results = []
    file_info = []
    # file_path = "C:/Users/jsilvarojas/source/repos/JIND-Multi/"
    file_path = "/app/"
    
    for file in files:
        if file.filename == '':
            print("archivo vacio")
            flash('No selected file')
            continue
        try:
            if file.filename.endswith('.h5ad'):
                print(file.filename)
                file_info.append({'filename': file.filename})
                file_path = os.path.join(file_path, file.filename)
                print(file_path)
                file.save(file_path)
                with open(file_json, 'r') as json_file:
                    try:
                        config = json.load(json_file)
                    except json.JSONDecodeError as e:
                        flash(f"Invalid JSON in file: {file_json} - {str(e)}")
                config['PATH'] = file_path
                with open(file_json, 'w') as json_file:
                        json.dump(config, json_file, indent=4)
        except Exception as e:
            flash(f"Error processing file {file.filename}: {str(e)}")
            results.append(f"Unexpected error: {str(e)}")
            continue

    return render_template('index.html', file_info=file_info, results=results)

@app.route("/test_emit")
def test_emit():
    socketio.emit('console_output', {'data': 'Test message from server'}, broadcast=True)
    return "Test message emitted"


@app.route("/run", methods=['POST'])
def run():
    """Executes the subprocess command and emits output via WebSockets."""
    # file_json = "config.json"
    command = ['run-jind-multi','--config',file_json]
    print(command)
    print("corriendo modelo")
    socketio.emit('console_output', {'data': "corriendo modelo\n"},broadcast=True)

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True
        )

        # Emit output from subprocess
        def emit_output(stream, label):
           for line in stream:
                # Asegúrate de emitir solo líneas no vacías
                if line.strip():
                    print(line, end='')  # Opcional: también imprime en la consola del servidor
                    socketio.emit('console_output', {'data': line})
                    socketio.sleep(0)
                socketio.sleep(0)
                
        # emit_output(process.stdout)
        # emit_output(process.stderr)
        # threading.Thread(target=emit_output, args=(process.stdout,), daemon=True).start()
        # threading.Thread(target=emit_output, args=(process.stderr,), daemon=True).start()
        socketio.start_background_task(emit_output, process.stdout, 'STDOUT')
        socketio.start_background_task(emit_output, process.stderr, 'STDERR')
        # process.stdout.close()
        # process.stderr.close()

        process.wait()

        return f"Command executed successfully: {process.returncode}"
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Error executing command: {e}\n"
            f"Return code: {e.returncode}\n"
            f"Command: {e.cmd}\n"
            f"Output: {e.output}\n"
            f"Error output: {e.stderr}\n"
        )
        flash(error_message)
    return redirect(url_for('home'))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True, port=5003)
