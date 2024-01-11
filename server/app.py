from flask import Flask, request
from flask_cors import CORS
import subprocess
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}})

user_replies = []
bot_replies = []

@socketio.on('connect')
def handle_connect():
    print('A client has connected to the server.')

@app.route("/message")
def handle_message():
    message = request.args.get('userInput')

    print("Received message...")
    userQuery = message
    print(userQuery)
    user_replies.append(userQuery)

    file_path = "text.txt"

    with open(file_path, "w") as file:
        file.writelines([userQuery])


    ml_file = "runner.py"    #correct it later
    subprocess.run(['python3', ml_file])

    row=""
    with open(file_path, "r") as file:
        row = file.read()
    bot_replies.append(row)
    print("Reply:",row)

    return row

@app.route("/feedback")
def handle_feedback():
    print("Getting feedback...")
    f_file = "feedback.py"
    f_value = request.args.get('value')
    print(f_value)
    with open(f_file, "w") as file:
        file.writelines([f_value, user_replies[-1], bot_replies[-1]])


if __name__ == '__main__':
    socketio.run(app, debug=True)
