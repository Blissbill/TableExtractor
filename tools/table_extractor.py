import logging
import re
from typing import List

import cv2
import numpy as np
import pytesseract
from easyocr import easyocr
from pyzbar.pyzbar import decode, ZBarSymbol

from tools.models import Rectangle, Cell, Table


READER = easyocr.Reader(['en', 'ru'], gpu=True)


def filter_duplicate_coordinates(rectangles: List[Rectangle], delta: int):
    remove_indexes = []
    for idx1, coord1 in enumerate(rectangles):
        for idx2 in range(idx1, len(rectangles)):
            coord2 = rectangles[idx2]
            if coord1.index < coord2.index and abs(coord1.top - coord2.top) <= delta \
                    and abs(coord1.left - coord2.left) <= delta and abs(coord1.width - coord2.width) <= delta \
                    and abs(coord1.height - coord2.height) <= delta:
                remove_indexes.append(idx2)
    logging.info(f"Found {len(remove_indexes)} duplicates")
    for idx in sorted(list(set(remove_indexes)), reverse=True):
        del rectangles[idx]
    return rectangles


def _get_parent_index(idx, parents_indexes):
    for i in parents_indexes:
        if idx == i[0]:
            return _get_parent_index(i[1], parents_indexes)
    return idx


def get_parent(rectangles: List[Rectangle]):
    parents_indexes = []
    for idx1, rect1 in enumerate(rectangles):
        for idx2, rect2 in enumerate(rectangles):
            if rect1.index != rect2.index and rect2.left <= rect1.left and \
                    (rect2.left + rect2.width) > (rect1.left + rect1.width) and \
                    rect2.top <= rect1.top and (rect2.top + rect2.height) > (rect1.top + rect1.height):
                parents_indexes.append((idx1, idx2))

    for idx in parents_indexes:
        rectangles[idx[0]].parent_index = rectangles[_get_parent_index(idx[1], parents_indexes)].index
    return rectangles


def get_parent1(rectangles: List[Rectangle], delta):
    parents_indexes = []
    for idx1, rect1 in enumerate(rectangles):
        for idx2, rect2 in enumerate(rectangles):
            if rect1.index != rect2.index and (rect2.left - delta) <= rect1.left and \
                    (rect2.left + rect2.width + delta) > (rect1.left + rect1.width) and \
                    (rect2.top - delta) <= rect1.top and (rect2.top + rect2.height + delta) > (rect1.top + rect1.height):
                parents_indexes.append((idx1, idx2))

    for idx in parents_indexes:
        rectangles[idx[0]].parent_index = idx[1]
    return rectangles


def clustering(rectangles: List[Rectangle], delta, comparator):
    clusters = {}
    for rect in rectangles:
        if rect.parent_index == -1:
            continue
        center = (rect.left + rect.width / 2, rect.top + rect.height / 2)
        added = False
        for cluster in clusters:
            if comparator(cluster, clusters[cluster], center, rect, delta):
                clusters[cluster].append(rect)
                added = True
                break
        if not added:
            clusters[center] = [rect]
    return clusters


def get_tables(rectangles: List[Rectangle], image: np.array):
    parent_indexes = []
    for rect in rectangles:
        if rect.parent_index != -1:
            parent_indexes.append(rect.parent_index)
    parent_idxes = list(set(parent_indexes))
    tables = []
    for rect in rectangles:
        if rect.index in parent_idxes:
            tables.append((image[rect.top: rect.top + rect.height, rect.left: rect.left + rect.width], rect))

    return tables


def processing_text(t: str):
    if t is None:
        return ""

    def o_to_zero(text: str):
        for zeros in re.finditer(r"(?<=\d)[oOоО]+", text):
            start_pos = zeros.span()[0]
            end_pos = zeros.span()[1]
            text = text[:start_pos] + "0" * (end_pos - start_pos) + text[end_pos:]
        return text

    def find_number_symbol(text: str):
        return re.sub("^n$|jg|ng", "№", text)

    def eng_to_rus(text: str):
        text = re.sub(r"(?<=[^\d])c(?=[^\d])", "с", text)
        text = re.sub(r"(?<=[^\d])e(?=[^\d])", "е", text)
        text = re.sub(r"(?<=[^\d])n(?=[^\d])", "п", text)
        text = re.sub(r"(?<=[^\d])b(?=[^\d])", "в", text)
        text = re.sub(r"(?<=[^\d])m(?=[^\d])", "м", text)
        text = re.sub(r"(?<=[^\d])t(?=[^\d])", "т", text)
        text = re.sub(r"(?<=[^\d])z(?=[^\d])", "г", text)
        return text

    def replace_by_part_text(text: str):
        text = re.sub(r"\sоля\s", " для ", text)
        text = re.sub(r"уолин", "удлин", text)
        text = re.sub(r"иермо", "термо", text)
        text = re.sub(r"испло|иепло|исило", "тепло", text)
        text = re.sub(r"изолямии", "изоляции", text)
        text = re.sub(r"еолов|солов", "голов", text)
        text = re.sub(r"еой", "вой", text)
        text = re.sub(r"сиае|сиав", "став", text)
        text = re.sub(r"еруз|арзу", "груз", text)
        text = re.sub(r"кламура", "клатура", text)
        text = re.sub(r"литания", "питания", text)
        text = re.sub(r"лоленциал", "потенциал", text)
        return text

    def find_units(text: str):
        text = re.sub(r"иn|wn|wm|lut", "шт", text)
        return text

    def find_three(text: str):
        text = re.sub(r"([зЗ](?=[,\d]))|((?<=[*xх])[зЗ])", "3", text)
        word_with_three = re.search(r"[^\w][зЗ]+[оО]*(кг|шт|л|г|мм)", text)
        if word_with_three:
            text = text[:word_with_three.span()[0]] + re.sub("[зЗ]", "3", word_with_three.group()) + text[word_with_three.span()[1]:]
        return text

    def find_six(text: str):
        text = re.sub(r"([бБ](?=[,\d]))|((?<=[*xх])[бБ])", "6", text)
        word_with_six = re.search(r"[^\w][бБ]+[оО]*(кг|шт|л|г|мм)", text)
        if word_with_six:
            text = text[:word_with_six.span()[0]] + re.sub("[бБ]", "6", word_with_six.group()) + text[
                                                                                                     word_with_six.span()[
                                                                                                         1]:]
        return text

    def find_one(text: str):
        text = re.sub(r"([lL](?=[,\d]))|((?<=[*xх])[lL])", "1", text)
        word_with_one = re.search(r"[^\w][lL]+[оО]*(кг|шт|л|г|мм)", text)
        if word_with_one:
            text = text[:word_with_one.span()[0]] + re.sub("[lL]", "1", word_with_one.group()) + text[
                                                                                                     word_with_one.span()[
                                                                                                         1]:]
        return text

    def fix_sign(text: str):
        text = re.sub(":$", ".", text)
        text = re.sub("@", "ф", text)
        return text

    t = t.strip().lower()
    if re.match(".*[а-яa-z].*", t):
        t = find_number_symbol(t)
        t = find_units(t)
        t = eng_to_rus(t)
        t = find_one(t)
        t = find_three(t)
        t = find_six(t)
        t = o_to_zero(t)
        t = replace_by_part_text(t)
        t = fix_sign(t)
        return t
    elif re.match(".*[\d].*", t):
        return "".join(re.findall("[\d.,]", t))
    else:
        return t


def parse_table(table: np.array, reader, ocr):
    logging.info("Parse table")
    preprocessed_table = preprocessing_image(table)
    contours, hierarchy = cv2.findContours(preprocessed_table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ogr = round(max(table.shape[0], table.shape[1]) * 0.01)
    delta = round(ogr / 2 + 0.5)
    rectangles = []
    idx = 0
    for i in range(0, len(contours)):
        l, t, w, h = cv2.boundingRect(contours[i])
        if h > ogr and w > ogr:
            rectangles.append(Rectangle(index=idx, left=l, top=t, width=w, height=h))
            idx += 1

    rectangles = get_parent1(rectangles, delta)
    rectangles = filter_duplicate_coordinates(rectangles, delta)
    main_parent = list(filter(lambda x: x.parent_index == -1, rectangles))[0]
    for i in range(len(rectangles) - 1, 0, -1):
        if rectangles[i].parent_index != main_parent.index:
            del rectangles[i]
    from tools.utils import draw_rects
    img = table.copy()
    draw_rects(img, rectangles)
    def column_comparator(cluster_key, cluster, center, rect, delta):
        if cluster_key[0] - delta <= center[0] <= cluster_key[0] + delta:
            maxW = min(cluster, key=lambda x: x.width).width
            if rect.width <= maxW * 1.3:
                return True
        return False

    def row_comparator(cluster_key, cluster, center, rect, delta):
        if cluster_key[1] - delta <= center[1] <= cluster_key[1] + delta:
            maxH = max(cluster, key=lambda x: x.height).height
            if rect.height <= maxH * 1.3:
                return True
        return False

    logging.info("Clustering column/rows")
    if len(rectangles) > 1000:
        logging.warning(f"Count rectangles over 1000 {len(rectangles)}")
        raise
    column_clusters = clustering(rectangles, delta, column_comparator)
    row_clusters = clustering(rectangles, delta, row_comparator)
    logging.info(f"count columns {len(column_clusters)}")
    logging.info(f"count rows {len(row_clusters)}")
    if len(column_clusters) > 20 or len(row_clusters) > 100:
        logging.warning(f"Count columns over 20 {len(column_clusters)} or count rows over 100 {len(row_clusters)}")
        raise

    drop_row_clusters = []
    for k in row_clusters:
        v = row_clusters[k]
        minY = min(v, key=lambda x: x.top).top
        maxY = max(v, key=lambda x: x.top + x.height)
        maxY = maxY.top + maxY.height

        if len(v) == 1:
            minX = min(v, key=lambda x: x.left).left
            maxX = max(v, key=lambda x: x.left + x.width)
            maxX = maxX.left + maxX.width
            if minX - delta <= 0 and maxX + delta >= table.shape[1]:
                drop_row_clusters.append(k)
                continue
        maxW = max(v, key=lambda x: x.width).width

        if maxW >= 0.6 * table.shape[1]:
            drop_row_clusters.append(k)
            continue

        for k1 in row_clusters:
            if k1 == k or k1 in drop_row_clusters: continue
            v1 = row_clusters[k1]
            minY1 = min(v1, key=lambda x: x.top).top
            maxY1 = max(v1, key=lambda x: x.top + x.height)
            maxY1 = maxY1.top + maxY1.height
            if minY - delta <= minY1 and maxY1 <= maxY + delta:
                row_clusters[k] += v1
                drop_row_clusters.append(k1)

    drop_column_clusters = []
    for k in column_clusters:
        v = column_clusters[k]
        minX = min(v, key=lambda x: x.left).left
        maxX = max(v, key=lambda x: x.left + x.width)
        maxX = maxX.left + maxX.width
        same_clusters = []
        for k1 in column_clusters:
            if k1 == k or k1 in drop_column_clusters: continue
            v1 = column_clusters[k1]
            minX1 = min(v1, key=lambda x: x.left).left
            maxX1 = max(v1, key=lambda x: x.left + x.width)
            maxX1 = maxX1.left + maxX1.width
            if minX - delta <= minX1 and maxX1 <= maxX + delta:
                same_clusters.append(k1)

        if len(same_clusters) > 0:
            drop_column_clusters.append(k)

        if len(same_clusters) > len(column_clusters.keys()) * (1.3 / 3):
            continue

        for c in same_clusters:
            column_clusters[c] += column_clusters[k]

    for drop in list(set(drop_row_clusters)):
        del row_clusters[drop]

    for drop in list(set(drop_column_clusters)):
        del column_clusters[drop]

    for k in column_clusters:
        column_clusters[k] = sorted(column_clusters[k], key=lambda x: x.top + x.height / 2)

    for k in row_clusters:
        row_clusters[k] = sorted(row_clusters[k], key=lambda x: x.left + x.width / 2)

    rows_centers = sorted(list(row_clusters.keys()), key=lambda x: x[1])
    table_object = Table()

    header_cluster = row_clusters[rows_centers[0]]
    logging.info("Make table object")
    logging.info("Make table header")

    margin = 3

    for column_idx, cl in enumerate(sorted(list(column_clusters.keys()))):
        cell = Cell(row=0, column=column_idx)
        text = ""
        for rect in list(set(column_clusters[cl]) & set(header_cluster)):
            img = table[rect.top + margin: rect.top + rect.height - margin,
                  rect.left + margin: rect.left + rect.width - margin]
            if ocr == "tesseract":
                txt = pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6')
            else:
                txt = reader.readtext(img, detail=False, link_threshold=0.1, text_threshold=0.25, low_text=0.3,
                                      min_size=1) or ""
            text = " ".join([text, txt])
        cell.text = processing_text(text)
        table_object.header.append(cell)
    logging.info("Make table body")
    for row_idx, cl in enumerate(rows_centers[1:]):
        row = []
        for column_idx, rect in enumerate(row_clusters[cl]):
            cell = Cell(row=row_idx, column=column_idx)
            img = table[rect.top + margin: rect.top + rect.height - margin,
                  rect.left + margin: rect.left + rect.width - margin]
            if ocr == "tesseract":
                txt = pytesseract.image_to_string(img, lang='rus+eng', config='--psm 6')
            else:
                txt = reader.readtext(img, detail=False, link_threshold=0.1, text_threshold=0.25, low_text=0.3,
                                      min_size=1) or ""
            cell.text = processing_text(txt)
            row.append(cell)
        try:
            table_object.add_row(row)
        except:
            if row_idx < len(rows_centers) - 2:
                raise
    logging.info("Table object was created successful")
    return table_object


def find_on_page(page_data, key, ocr):
    logging.info(f"Find [{key}] on page")
    found_key = None
    for idx, data in enumerate(page_data):
        if re.search(key, data[1].lower().strip(), flags=re.IGNORECASE):
            found_key = data
            break
    if found_key is None:
        return None
    key_center = (found_key[0][0][0] + (found_key[0][2][0] - found_key[0][0][0]) // 2, found_key[0][0][1] + (found_key[0][2][1] - found_key[0][0][1]) // 2)
    key_height = found_key[0][2][1] - found_key[0][0][1]
    result = ""
    for idx, data in enumerate(page_data):
        center = (data[0][0][0] + (data[0][2][0] - data[0][0][0]) // 2, data[0][0][1] + (data[0][2][1] - data[0][0][1]) // 2)
        if key_center[1] - key_height <= center[1] <= key_center[1] + key_height:
            result += " " + data[1]
    return result.strip() if result != "" else None


def preprocessing_image(image, find_qr=False):
    img = image.copy()

    if find_qr:
        for dec in decode(img, symbols=[ZBarSymbol.QRCODE]):
            if dec.type == "QRCODE":
                img[dec.rect.top:dec.rect.top + dec.rect.height, dec.rect.left:dec.rect.left + dec.rect.width] = [255, 255, 255]

    gray_image = img[:, :, 0]
    thresh_value = cv2.adaptiveThreshold(cv2.GaussianBlur(gray_image, (7, 7), 0), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)
    return cv2.GaussianBlur(thresh_value, (3, 3), 0)


def extract_tables(image, document_info={}, ocr="tesseract"):
    logging.info(f"Extract table start")
    preprocessed_image = preprocessing_image(image, True)
    contours, hierarchy = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ogr = round(max(image.shape[0], image.shape[1]) * 0.01)
    delta = round(ogr / 2 + 0.5)
    ind = 1
    rectangles = []
    for i in range(0, len(contours)):
        l, t, w, h = cv2.boundingRect(contours[i])
        if h > ogr and w > ogr:
            rectangles.append(Rectangle(index=ind, left=l, top=t, width=w, height=h))
            ind = ind + 1

    rectangles = filter_duplicate_coordinates(rectangles, delta)
    rectangles = get_parent(rectangles)
    logging.info(f"Find tables")
    tables_images = get_tables(rectangles, image)

    addition_info = {}
    if document_info:
        copy_image = image.copy()
        for idx, table in enumerate(tables_images):
            rect = table[1]
            copy_image[rect.top: rect.top + rect.height, rect.left: rect.left + rect.width] = 255
        text_from_image = READER.readtext(copy_image, paragraph=True, x_ths=0.3, y_ths=0.3)
        # text_from_image = pytesseract.image_to_data(copy_image, lang='rus+eng', config="--psm 6", output_type=pytesseract.Output.DICT)

        for info in document_info["expressions"]:
            addition_info[info["key"]] = processing_text(find_on_page(text_from_image, info["regex"], ocr))
    tables = []

    for i, table in enumerate(tables_images):
        cv2.imwrite(f"tables/table-{i}.jpg", table[0])

    for table_idx, table in enumerate(tables_images):
        try:
            tables.append(parse_table(table[0], READER, ocr))
        except Exception as e:
            import traceback
            logging.error(f"extract_tables::error {traceback.format_exc()}")
            continue
    return tables, addition_info




