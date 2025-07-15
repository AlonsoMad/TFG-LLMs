from flask import Flask, make_response, request# type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore
from flask_login import LoginManager# type: ignore
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

db = SQLAlchemy()
DB_NAME = os.getenv("DB_NAME", "/db_data/database_tfg.db")

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="../app/static")
    app.config['SECRET_KEY'] = os.getenv("WEB_APP_KEY")

    db_path = DB_NAME
    print(f"Database path: {db_path}")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    db.init_app(app)
    CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

    from views import views
    from auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from models import User
    with app.app_context():
        create_database(app, db_path)

    @app.after_request
    def add_cache_control(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    @login_manager.user_loader
    def load_user(user_id):
        from models import User
        return User.query.get(int(user_id))
    
    
    return app

def create_database(app, db_path):
    print(db_path)
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.create_all()
        print(f"Created database {db_path}")

