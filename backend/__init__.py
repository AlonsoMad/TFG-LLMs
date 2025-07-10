from flask import Flask
from dotenv import load_dotenv
import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

load_dotenv()  # Load environment variables from .env file

db = SQLAlchemy()
DB_NAME = os.getenv("DB_NAME", "database_tfg.db")

def create_app():
    app = Flask(__name__, template_folder="../app/templates", static_folder="../app/static")
    app.config['SECRET_KEY'] = os.getenv("WEB_APP_KEY")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)


    from app.views import views
    from app.auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from backend.models import User
    with app.app_context():
        create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    @login_manager.user_loader
    def load_user(user_id):
        from backend.models import User
        return User.query.get(int(user_id))
    
    
    return app

def create_database(app):
    if not os.path.exists(DB_NAME):
        db.create_all()
        print(f"Created database {DB_NAME}")