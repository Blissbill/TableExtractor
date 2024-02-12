import logging
import os
import re
import tempfile

import openai
import requests
from flask import Blueprint, request, current_app, jsonify

from api.settings import Config


blueprint = Blueprint('text_extractor', __name__, url_prefix='/text_extractor')

openai.api_key = Config.GPT_KEY
PROMPT = ("'{RAW_TEXT}'"
          "Приведи его в табличный вид в формате: | Название | Единицы измерения | Количество |"
          "Кроме таблицы больше ничего не пиши"
          "Не пытайся написать скрипт на python, используй только GPT)"
          "Текст может содержать адреса, игнорируй их"
          "На выход не выдавай ничего, кроме таблицы"
          "Если единицы - Штуки - в таблице пиши 'шт', мешки - 'меш'"
          "Исправляй правописание"
          "Используй именительный падеж"
          "Саморез металл - это саморез по металлу"
          "Дюбель гвоздь, а не дюпель"
          "Шумка - это шумоизоляция")


def text_to_list_chat_gpt(text, prompt):
    if prompt is None:
        prompt = PROMPT
    messages = [{"role": "user", "content": prompt.format(RAW_TEXT=text)}]
    logging.info("call gpt")
    response = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=messages)
    content = response.choices[0].message.content
    result = []
    skip_lines = 2
    for idx, line in enumerate(content.split("\n")):
        if not line.strip() or idx + 1 <= skip_lines:
            continue
        items = line.split("|")
        name = items[1]
        name = re.sub("^\d+.\s*", "", name).strip()
        product = {
            "name": name,
            "unit": None,
            "count": None
        }
        if len(items) > 2:
            sizes = re.sub("\s*-*\s*", "", items[2]).strip()
            if sizes:
                product["unit"] = sizes
        if len(items) > 3:
            count = re.sub("\s*-*\s*", "", items[3]).strip()
            if count:
                product["count"] = count
        result.append(product)
    return result


@blueprint.route('/image', methods=['POST'])
def image_extract():
    tmp_file = tempfile.NamedTemporaryFile(dir=current_app.config["TMP_FOLDER"], suffix=".jpg", delete=False)
    with open(tmp_file.name, "wb") as f:
        f.write(request.files["image"].read())
    with open(tmp_file.name, "rb") as f:
        response = requests.post(Config.REHAND_HOST, files={
            "file": (f.name, f.read())
        }, data={
            "type": Config.TYPE_VALUE
        },  headers={"Authorization": Config.REHAND_API_KEY})
        response.raise_for_status()
    try:
        os.remove(tmp_file.name)
    except:
        pass

    return jsonify(text_to_list_chat_gpt(response.json()["output_text"]))


@blueprint.route('/text', methods=['POST'])
def text_extract():
    logging.info("table_extractor::post")
    return jsonify(text_to_list_chat_gpt(request.get_json()["text"], prompt=request.get_json().get("prompt")))


@blueprint.route('/text/mock', methods=['POST'])
def text_extract_mock():
    return jsonify([{"name": "Позиция 1", "count": "1", "unit": "кг"},
                    {"name": "Позиция 2", "count": "2", "unit": "шт"},
                    {"name": "Позиция 3", "count": "3", "unit": "кг"}])
