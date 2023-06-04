import os
import tempfile

from flask import Blueprint, request, current_app, jsonify

from tools.table_extractor import extract_tables
from tools.utils import pdf_to_images

blueprint = Blueprint('table_extractor', __name__, url_prefix='/table_extractor')


@blueprint.route('/pdf', methods=['POST'])
def pdf_table_extract():
    result = {}
    with tempfile.NamedTemporaryFile(dir=current_app.config["TMP_FOLDER"], suffix=".pdf", delete=False) as f:
        f.write(request.files["pdf"].read())
        pdf_name = f.name
    try:

        for page_idx, page_img in enumerate(pdf_to_images(pdf_name)):
            print(f"Page {page_idx} started...")
            result[f"page_{page_idx}"] = {}
            extra_info = []
            if page_idx == 0:
                extra_info = ["поставщик", "покупатель", "счет на оплату"]
            tables, addition_info = extract_tables(page_img, extra_info)
            result[f"page_{page_idx}"] = tables
            result.update(addition_info)
    except Exception:
        pass
    finally:
        os.remove(pdf_name)
    with open("response.json", "w", encoding="utf-8") as file:
        import json
        json.dump(result, file, ensure_ascii=False, indent=4)
    return jsonify(result)
