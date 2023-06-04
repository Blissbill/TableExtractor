from api.app import create_app
from api.settings import Config

application = create_app(Config)

if __name__ == "__main__":
    application.run(port=8555)
