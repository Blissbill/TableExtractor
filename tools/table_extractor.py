from typing import List

import cv2
import numpy as np
from easyocr import easyocr

from tools.models import Rectangle, Cell, Table


READER = easyocr.Reader(['en', 'ru'])


def filter_duplicate_coordinates(rectangles: List[Rectangle], delta: int):
    remove_indexes = []
    for idx1, coord1 in enumerate(rectangles):
        for idx2 in range(idx1, len(rectangles)):
            coord2 = rectangles[idx2]
            if coord1.index < coord2.index and abs(coord1.top - coord2.top) <= delta \
                    and abs(coord1.left - coord2.left) <= delta and abs(coord1.width - coord2.width) <= delta \
                    and abs(coord1.height - coord2.height) <= delta:
                remove_indexes.append(idx2)
    print(f"Found {len(remove_indexes)} duplicates")
    for idx in sorted(remove_indexes, reverse=True):
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


def parse_table(table: np.array, reader):
    print("Parse table")
    preprocessed_table = preprocessing_image(table)
    contours, hierarchy = cv2.findContours(preprocessed_table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ogr = round(max(table.shape[0], table.shape[1]) * 0.013)
    delta = round(ogr / 2 + 0.5)
    rectangles = []
    idx = 0
    for i in range(0, len(contours)):
        l, t, w, h = cv2.boundingRect(contours[i])
        if h > ogr and w > ogr:
            rectangles.append(Rectangle(index=idx, left=l, top=t, width=w, height=h))
            idx += 1
    rectangles = get_parent(rectangles)
    rectangles = filter_duplicate_coordinates(rectangles, delta)


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

    print("Clustering column/rows")
    column_clusters = clustering(rectangles, delta, column_comparator)
    row_clusters = clustering(rectangles, delta, row_comparator)

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

    rows_centers = sorted(list(row_clusters.keys()))
    table_object = Table()

    header_cluster = row_clusters[rows_centers[0]]
    print("Make table object")
    print("Make table header")
    for column_idx, cl in enumerate(sorted(list(column_clusters.keys()))):
        cell = Cell(row=0, column=column_idx)
        text = ""
        for rect in list(set(column_clusters[cl]) & set(header_cluster)):
            img = table[rect.top: rect.top + rect.height, rect.left: rect.left + rect.width]
            txt = reader.readtext(img, detail=False)
            if txt:
                text += txt[0]
        cell.text = text
        table_object.header.append(cell)
    print("Make table body")
    for row_idx, cl in enumerate(rows_centers[1:]):
        row = []
        for column_idx, rect in enumerate(row_clusters[cl]):
            cell = Cell(row=row_idx, column=column_idx)
            img = table[rect.top: rect.top + rect.height, rect.left: rect.left + rect.width]
            r = reader.readtext(img, detail=False, text_threshold=0.3, low_text=0.3)
            if len(r):
                cell.text = r[0]
            row.append(cell)
        table_object.add_row(row)
    print("Table object was created successful")
    return table_object.to_dict()


def find_on_page(page_data, key):
    print(f"Find [{key}] on page")
    found_key = None
    for data in page_data:
        if key in data[1].lower():
            found_key = data
            break
    if found_key is None:
        return None
    key_center = (found_key[0][0][0] + (found_key[0][2][0] - found_key[0][0][0]) // 2, found_key[0][0][1] + (found_key[0][2][1] - found_key[0][0][1]) // 2)
    key_height = found_key[0][2][1] - found_key[0][0][1]
    result = ""
    for data in page_data:
        center = (data[0][0][0] + (data[0][2][0] - data[0][0][0]) // 2, data[0][0][1] + (data[0][2][1] - data[0][0][1]) // 2)
        if key_center[1] - key_height <= center[1] <= key_center[1] + key_height:
            result += " " + data[1]

    return result if result != "" else None


def preprocessing_image(image):
    img = image.copy()
    gray_image = img[:, :, 0]
    ret, thresh_value = cv2.threshold(gray_image, 75, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    obr_img = cv2.erode(thresh_value, kernel, iterations=1)
    return cv2.GaussianBlur(obr_img, (3, 3), 0)


def extract_tables(image, extra_info=[]):
    print(f"Extract table start")
    preprocessed_image = preprocessing_image(image)
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
    print(f"Find tables")
    tables_images = get_tables(rectangles, image)

    addition_info = {}
    if extra_info:
        copy_image = image.copy()
        for idx, table in enumerate(tables_images):
            rect = table[1]
            copy_image[rect.top: rect.top + rect.height, rect.left: rect.left + rect.width] = 255
        text_from_image = READER.readtext(copy_image)
        for key in extra_info:
            addition_info[key] = find_on_page(text_from_image, key)
    tables = {}
    for table_idx, table in enumerate(tables_images):
        try:
            tables[f"table_{table_idx}"] = parse_table(table[0], READER)
        except Exception:
            continue
    return tables, addition_info




