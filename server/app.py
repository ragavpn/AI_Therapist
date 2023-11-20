from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import subprocess
# from pydub import AudioSegment
from decouple import config
import speech_recognition as sr
import time
# import base64

from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
recognizer = sr.Recognizer()
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}})

# db_data = {
#     'host': config('DB_HOST'),
#     'user': config('DB_USER'),
#     'password': config('DB_PASSWORD'),
#     'database': config('DB_NAME')
# }

# con = mysql.connector.connect(host='localhost', user='root', password='!Ndr0med1')
# con = mysql.connector.connect(**db_data)
# print(con)
# cursor = con.cursor()
# create_statement = """CREATE TABLE IF NOT EXISTS users (
#         user_id INT AUTO_INCREMENT PRIMARY KEY,
#         fullName VARCHAR(50) NOT NULL,
#         email VARCHAR(50) NOT NULL UNIQUE,
#         password VARCHAR(30) NOT NULL
#     )"""
# cursor.execute(create_statement)
# con.commit()
# print("Created...")

user_replies = []
bot_replies = []

# @app.route("/api/get-audio", methods=['POST'])
# def get_data():

#     query_data = request.json.get('textData')

#     placeholder_output = "I understand that feeling."

#     chat_history.append({'user': query_data, 'chatbot': placeholder_output})

#     emit('update_messages', chat_history, broadcast=True)

#     print(query_data)
#     print("Query received")

#     file_path = "output.txt"

#     with open(file_path, "a") as file:
#         file_path.write(query_data)

#     response = {
#         'message': f'Received data'
#     }

#     return jsonify(response)
    
@socketio.on('connect')
def handle_connect():
    print('A client has connected to the server.')


# @socketio.on('message')
@app.route("/message")
def handle_message():
    message = request.args.get('userInput')

    # print(message)
    print("Received message...")
    userQuery = message
    print(userQuery)
    user_replies.append(userQuery)

    file_path = "text.txt"
    # placeholder_output = "I understand that feeling."

    with open(file_path, "w") as file:
        file.writelines([userQuery])

    # with open(file_path, "w") as file:
    #     file.writelines([placeholder_output])
        

    ml_file = "runner.py"    #correct it later
    subprocess.run(['python3', ml_file])

    # time.sleep(5)

    row=""
    with open(file_path, "r") as file:
        row = file.read()
    bot_replies.append(row)
    print("Reply:",row)
    # socketio.emit('message', row)
    return row

# @socketio.on('feedback')
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
