import os

from flask import Flask

from api import table_extractor, services


def create_app(config_object):
    app = Flask(__name__.split('.')[0])
    app.config.from_object(config_object)
    app.url_map.strict_slashes = False
    register_blueprints(app)
    create_folders(app)
    return app


def register_blueprints(app):
    app.register_blueprint(table_extractor.views.blueprint)
    app.register_blueprint(services.views.blueprint)
    return None


def create_folders(app):
    os.makedirs(app.config["TMP_FOLDER"], exist_ok=True)
