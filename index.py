import os
import sys
import time
import json
import secrets
import subprocess
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_socketio import SocketIO,emit
from flask_bootstrap import Bootstrap5

app = Flask(__name__)
secret_key = secrets.token_hex(32)
print("Your secret key:", secret_key)
app.secret_key = secret_key
bootstrap = Bootstrap5(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# class ConsoleLogger:
#     def __init__(self, socketio):
#         self.socketio = socketio

#     def write(self, message):
#         if message.strip():  # Evita enviar líneas vacías
#             self.socketio.emit('console_output', {'data': message})

#     def flush(self):
#         pass

@app.route("/", methods=['GET', 'POST'])
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
    file_path = "C:/Users/jsilvarojas/source/repos/JIND-Multi/"
    file_json = "config.json"
    
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

        except Exception as e:
            flash(f"Error processing file {file.filename}: {str(e)}")
            results.append(f"Unexpected error: {str(e)}")
            continue

    return render_template('index.html', file_info=file_info, results=results)

@app.route("/run", methods=['GET','POST'])
def run():
    """Executes the subprocess command and emits output via WebSockets."""
    file_json = "config.json"
    command = ['run-jind-multi', '--config', file_json]
    print(command)
    print("corriendo modelo")
    socketio.emit('console_output', {'data': "corriendo modelo\n"},broadcast=True)

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True
        )

        # Emit output from subprocess
        def emit_output(stream):
           for line in stream:
                # Asegúrate de emitir solo líneas no vacías
                if line.strip():
                    print(line, end='')  # Opcional: también imprime en la consola del servidor
                    socketio.emit('console_output', {'data': line})
                    socketio.sleep(0)
                socketio.sleep(0)

        emit_output(process.stdout)
        emit_output(process.stderr)

        process.stdout.close()
        process.stderr.close()
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
    socketio.run(app, debug=True, port=5003)
