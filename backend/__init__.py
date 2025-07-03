from flask import Flask

def create_app():
    app = Flask(__name__, template_folder="../app/templates", static_folder="../app/static")
    app.config['SECRET_KEY'] = 'OPENAI_GPT4O_KEY'# TODO: Pregunta lorena q se pone aqui 'your_secret_key_here'

    from app.views import views
    from app.auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    return app