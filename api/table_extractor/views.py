import os
import re
import tempfile

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
    try:
        table_idx = 0
        for page_idx, page_img in enumerate(pdf_to_images(pdf_name)):
            print(f"Page {page_idx} started...")
            extra_info = []
            if page_idx == 0:
                extra_info = ["поставщик:", "покупатель:", "счет на оплату"]
            tables, addition_info = extract_tables(page_img, extra_info)
            for t in tables:
                result["tables"][f"table_{table_idx}"] = t
                table_idx += 1
            for ai, v in addition_info.items():
                addition_info[ai] = v.lower().replace(ai, "").strip()

            for info in ["счет на оплату"]:
                if addition_info.get(info):
                    result["document_type"] = info
                    date = re.search(r"(\d{2}\.){2}\d{4}", addition_info[info].lower(), flags=re.IGNORECASE)
                    if date:
                        result["date"] = date.group(0)
                    else:
                        date = re.search(r"(?P<day>\d{2})\s(?P<month>(янв|фев|март|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*)\s(?P<year>\d{4})", addition_info[info].lower(), flags=re.IGNORECASE)
                        if date:
                            result["date"] = f"{date.group('day')}.{month_mapping(date.group('month'))}.{date.group('year')}"
            addition_info.pop("счет на оплату")
            result.update(addition_info)
    except Exception as e:
        pass
    finally:
        os.remove(pdf_name)
    return jsonify(result)
