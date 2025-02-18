from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///backend/database/attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)

from routes.employees import employees_bp
from routes.attendance import attendance_bp

app.register_blueprint(employees_bp, url_prefix="/employees")
app.register_blueprint(attendance_bp, url_prefix="/attendance")

if __name__ == "__main__":
    app.run(debug=True)
