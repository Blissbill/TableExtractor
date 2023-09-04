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
            cv2.imwrite("tmp.jpg", page_img)
            extra_info = []
            if page_idx == 0:
                extra_info = ["(поставщик|исполнитель):", "(окупатель|заказчик):", "счет.*[N|Ng|№].*от"]
            tables, addition_info = extract_tables(page_img, extra_info, ocr)
            all_tables += tables
            if page_idx == 0:
                for ai, v in addition_info.items():
                    if v:
                        addition_info[ai] = re.sub(ai, "", v.lower()).strip()
                for info in ["счет.*[N|Ng|№].*от"]:
                    if addition_info.get(info):
                        result["document_type"] = "счет на оплату"
                        date = re.search(r"(\d{2}\.){2}\d{4}", addition_info[info].lower(), flags=re.IGNORECASE)
                        if date:
                            result["date"] = date.group(0)
                        else:
                            date = re.search(
                                r"(?P<day>\d{2})\s(?P<month>(янв|фев|март|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*)\s(?P<year>\d{4})",
                                addition_info[info].lower(), flags=re.IGNORECASE)
                            if date:
                                result[
                                    "date"] = f"{date.group('day')}.{month_mapping(date.group('month'))}.{date.group('year')}"
                        result["№"] = None
                        document_number = re.search("(?<=[№|N] ).*? ", addition_info[info], flags=re.IGNORECASE)
                        if document_number:
                            result["№"] = document_number.group(0).strip()
                addition_info.pop("счет.*[N|Ng|№].*от")
                addition_info["поставщик"] = addition_info.pop("(поставщик|исполнитель):")
                addition_info["покупатель"] = addition_info.pop("(окупатель|заказчик):")
                result.update(addition_info)

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
    extra_info = ["(поставщик|исполнитель):", "(окупатель|заказчик):", "счет.*[N|Ng|№].*от"]
    with tempfile.NamedTemporaryFile(dir=current_app.config["TMP_FOLDER"], suffix=".jpg", delete=False) as f:
        f.write(request.files["image"].read())
        page_img = cv2.imread(f.name)
    ocr = request.form.get("ocr") or "tesseract"
    tables, addition_info = extract_tables(page_img, extra_info, ocr)
    for ai, v in addition_info.items():
        if v:
            addition_info[ai] = re.sub(ai, "", v.lower()).strip()
    for info in ["счет.*[N|Ng|№].*от"]:
        if addition_info.get(info):
            result["document_type"] = "счет на оплату"
            date = re.search(r"(\d{2}\.){2}\d{4}", addition_info[info].lower(), flags=re.IGNORECASE)
            if date:
                result["date"] = date.group(0)
            else:
                date = re.search(
                    r"(?P<day>\d{2})\s(?P<month>(янв|фев|март|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*)\s(?P<year>\d{4})",
                    addition_info[info].lower(), flags=re.IGNORECASE)
                if date:
                    result[
                        "date"] = f"{date.group('day')}.{month_mapping(date.group('month'))}.{date.group('year')}"
            result["№"] = None
            document_number = re.search("(?<=[№|N] ).*? ", addition_info[info], flags=re.IGNORECASE)
            if document_number:
                result["№"] = document_number.group(0).strip()
    addition_info.pop("счет.*[N|Ng|№].*от")
    addition_info["поставщик"] = addition_info.pop("(поставщик|исполнитель):")
    addition_info["покупатель"] = addition_info.pop("(окупатель|заказчик):")
    result.update(addition_info)
    for idx, t in enumerate(tables):
        result["tables"][f"table_{idx}"] = t.to_dict()
    return jsonify(result)
