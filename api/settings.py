import os
from dotenv import load_dotenv

load_dotenv()


class Config(object):
    DEBUG = True
    TMP_FOLDER = "./tmp/"
    REHAND_HOST = "https://rehand.ru/api/v1/upload"
    REHAND_API_KEY = os.environ.get("REHAND_API_KEY")
    TYPE_VALUE = "handwriting"
    GPT_KEY = os.environ.get("GPT_KEY")
