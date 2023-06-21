import argparse
import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_message():
    message = request.get_data(as_text=True)
    print(f'Received message: {message}')
    return 'OK'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A command line tool for reading local files and sending/receiving messages to/from a web server.')
    parser.add_argument('filename', type=str, help='The name of the file to read.')
    args = parser.parse_args()

    filepath = os.path.abspath(args.filename)
    with open(filepath, 'r') as f:
        contents = f.read()
        print(f'Read file {filepath}:')
        print(contents)

    app.run()

# python3 sdk_cli.py tmp.txt
# curl -X POST -d "Hello, world!" http://localhost:5000/