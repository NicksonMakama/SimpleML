from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def sendData():
    d={}
    d['Query'] = str(request.args['Query'])
    return jsonify(d)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
