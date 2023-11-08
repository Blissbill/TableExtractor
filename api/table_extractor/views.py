import json
import os
import logging
import re
import tempfile
import cv2

from flask import Blueprint, request, current_app, jsonify

from tools.table_extractor import extract_tables
from tools.utils import pdf_to_images

blueprint = Blueprint('table_extractor', __name__, url_prefix='/table_extractor')


def month_mapping(month):
    for m in (("янв", "01"), ("фев", "02"), ("мар", "03"), ("апр", "04"), ("ма", "05"), ("июн", "06"), ("июл", "07"),
              ("авг", "08"), ("сен", "09"), ("окт", "10"), ("ноя", "11"), ("дк", "12")):
        if m[0] in month.lower():
            return m[1]
    return month


@blueprint.route('/pdf', methods=['POST'])
def pdf_table_extract():
    result = {"tables": {}, "document_type": None, "date": None}
    with tempfile.NamedTemporaryFile(dir=current_app.config["TMP_FOLDER"], suffix=".pdf", delete=False) as f:
        f.write(request.files["pdf"].read())
        pdf_name = f.name
    ocr = request.form.get("ocr") or "tesseract"
    try:
        table_idx = 0
        all_tables = []
        merge_tables = {}
        used_tables = []

        for page_idx, page_img in enumerate(pdf_to_images(pdf_name)):
            logging.info(f"Page {page_idx} started...")
            document_info = {}
            if page_idx == 0:
                with open("document_info.json", encoding="utf-8") as f:
                    document_info = json.load(f)
            tables, addition_info = extract_tables(page_img, document_info, ocr)
            all_tables += tables
            if page_idx == 0:
                for di in document_info["expressions"]:
                    if addition_info.get(di["key"]):
                        if di.get("expressions"):
                            result["document_type"] = "счет на оплату"
                            for ii in di["expressions"]:
                                data = re.search(ii["regex"], addition_info[di["key"]].lower(), flags=re.IGNORECASE)
                                if data:
                                    if ii["key"] == "date":
                                        result[ii["key"]] = ii["data_format"].format(*[month_mapping(data.group(g).strip()) for g in ii["regex_groups"]])
                                    else:
                                        result[ii["key"]] = ii["data_format"].format(
                                            *[data.group(g).strip() for g in ii["regex_groups"]])
                        else:
                            result[di["key"]] = re.sub(di["regex"], "", addition_info[di["key"]].lower()).strip()
                    else:
                        result[di["key"]] = None
        for idx1, t1 in enumerate(all_tables):
            if idx1 in used_tables or t1.is_empty(): continue
            merge_tables[idx1] = []
            for idx2, t2 in enumerate(all_tables):
                if idx1 == idx2 or len(t1.header) != len(t2.header) or idx2 in used_tables: continue
                count_same_values = 0
                for i in range(0, len(t1.header)):
                    if t1.header[i].text.lower() == t2.header[i].text.lower():
                        count_same_values += 1
                if count_same_values >= len(t1.header) // 2:
                    merge_tables[idx1].append(t2)
                    used_tables.append(idx2)
            used_tables.append(idx1)
        updated_tables = []

        for k, v in merge_tables.items():
            for t in v:
                for row in t:
                    all_tables[k].add_row(row)
            updated_tables.append(all_tables[k])
        for t in updated_tables:
            result["tables"][f"table_{table_idx}"] = t.to_dict()
            table_idx += 1
    except Exception as e:
        import traceback
        logging.error(f"extract_tables::error {traceback.format_exc()}")
    finally:
        os.remove(pdf_name)
    return jsonify(result)


@blueprint.route('/image', methods=['POST'])
def image_table_extractor():
    result = {"tables": {}, "document_type": None, "date": None}
    with tempfile.NamedTemporaryFile(dir=current_app.config["TMP_FOLDER"], suffix=".jpg", delete=False) as f:
        f.write(request.files["image"].read())
        page_img = cv2.imread(f.name)
    ocr = request.form.get("ocr") or "tesseract"
    with open("document_info.json", encoding="utf-8") as f:
        document_info = json.load(f)
    tables, addition_info = extract_tables(page_img, document_info, ocr)
    for di in document_info["expressions"]:
        if addition_info.get(di["key"]):
            if di.get("expressions"):
                result["document_type"] = "счет на оплату"
                for ii in di["expressions"]:
                    data = re.search(ii["regex"], addition_info[di["key"]].lower(), flags=re.IGNORECASE)
                    if data:
                        if ii["key"] == "date":
                            result[ii["key"]] = ii["data_format"].format(
                                *[month_mapping(data.group(g).strip()) for g in ii["regex_groups"]])
                        else:
                            result[ii["key"]] = ii["data_format"].format(
                                *[data.group(g).strip() for g in ii["regex_groups"]])
            else:
                result[di["key"]] = re.sub(di["regex"], "", addition_info[di["key"]].lower()).strip()
    for idx, t in enumerate(tables):
        result["tables"][f"table_{idx}"] = t.to_dict()
    return jsonify(result)
