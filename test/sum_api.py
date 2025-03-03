from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/sum', methods=['POST'])
def sum_numbers():
    data = request.json  # Nhận dữ liệu từ frontend
    num1 = data.get("num1")
    num2 = data.get("num2")

    if num1 is None or num2 is None:
        return jsonify({"error": "Missing numbers"}), 400

    result = num1 + num2
    return jsonify({"sum": result})

if __name__ == '__main__':
    app.run(debug=True)