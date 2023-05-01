import math
import random
from typing import Iterator, List

import fitz  # noqa
import numpy as np
import cv2

# from Recognize import recognzie
from models import Rectangle, Cell, Table


def draw_rects(image: np.array, coordinates: List[Rectangle], color=(0, 255, 0), thickness=2):
    for coord in coordinates:
        if coord.parent_index == -1:
            cv2.rectangle(image, (coord.left, coord.top), (coord.left + coord.width, coord.top + coord.height),
                          (255, 0, 0),
                          1)
            cv2.putText(image, str(coord.index), (coord.left, coord.top + coord.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (coord.left, coord.top), (coord.left + coord.width, coord.top + coord.height), color,
                          thickness)
            cv2.putText(image, str(coord.index), (coord.left, coord.top + coord.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)


def pdf_to_images(path: str) -> Iterator[np.ndarray]:
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        yield np.ascontiguousarray(image[..., [2, 1, 0]])


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
        # rectangles[idx[0]].parent_index = rectangles[idx[1]].index
    return rectangles


def new_neighboring_nodes(pairs):
    a = []
    for p1 in pairs:
        for p2 in pairs:
            if p1[1] == p2[0] and p1[0] < p2[0]:
                a.append((p1[0], p2[1]))
    a = list(set(a))
    b = []
    for p1 in pairs:
        for p2 in pairs:
            if p1[1] == p2[1] and p1[0] < p2[0]:
                b.append((p1[0], p2[0]))
    b = list(set(b))

    return list(set(a + b))


def removed_neighbors(pairs):
    removes = []
    a = []
    for i, p1 in enumerate(pairs):
        for idx1, p2 in enumerate(pairs):
            if p1[0] == p2[1] and p1[0] > p2[0]:
                for idx2, p3 in enumerate(pairs):
                    if p1[1] == p3[1] and p1[0] > p2[0] and p2[0] == p3[0]:
                        removes.append(p1)
                        break
    return list(set(removes))



def create_blocks(rectangles: List[Rectangle], delta):
    pairs_indexes = []
    for idx1, rect1 in enumerate(rectangles):
        for idx2, rect2 in enumerate(rectangles):
            if rect1.index < rect2.index and rect1.parent_index == rect2.parent_index and rect1.parent_index != -1 and (
                (
                    (abs(rect1.left + rect1.width - rect2.left) <= delta or abs(rect2.left + rect2.width - rect1.left) <= delta)
                    and
                    (
                        (rect1.top <= rect2.top and rect2.top <= rect1.top + rect1.height and rect1.top + rect1.height <= rect2.top + rect2.height)
                        or
                        (rect2.top <= rect1.top and rect1.top <= rect2.top + rect2.height and rect2.top + rect2.height <= rect1.top + rect1.height)
                        or
                        (rect2.top <= rect1.top and rect1.top + rect1.height <= rect2.top + rect2.height)
                        or
                        (rect1.top <= rect2.top and rect2.top + rect2.height <= rect1.top + rect1.height)
                    )
                )
                or
                (
                    (abs(rect1.top + rect1.height - rect2.top) <= delta or abs(
                        rect2.top + rect2.height - rect1.top) <= delta)
                    and
                    (
                        (rect1.left <= rect2.left and rect2.left <= rect1.left + rect1.width and rect1.left + rect1.width <= rect2.left + rect2.width)
                        or
                        (rect2.left <= rect1.left and rect1.left <= rect2.left + rect2.width and rect2.left + rect2.width <= rect1.left + rect1.width)
                        or
                        (rect2.left <= rect1.left and rect1.left + rect1.width <= rect2.left + rect2.width)
                        or
                        (rect1.left <= rect2.left and rect2.left + rect2.width <= rect1.left + rect1.width)
                    )
                )
            ):
                for idx3, rect3 in enumerate(rectangles):
                    if rect2.parent_index == rect3.index and rect3.parent_index == -1:
                        pairs_indexes.append((rect1.index, rect2.index))

    while True:
        new_neighbors = new_neighboring_nodes(pairs_indexes)
        pairs_indexes += new_neighbors
        pairs_indexes = list(set(pairs_indexes))
        removed = removed_neighbors(pairs_indexes)
        if not removed:
            break
        for remove in removed:
            del pairs_indexes[pairs_indexes.index(remove)]

    some_pairs = []
    for pair in pairs_indexes:
        some_pairs.append((pair[0], pair[0]))

    pairs_indexes = list(set(pairs_indexes + some_pairs))

    for rect in rectangles:
        for pair in pairs_indexes:
            if rect.index == pair[1]:
                rect.block_index = pair[0]

    return rectangles


def remove_trash(rectangles: List[Rectangle]):
    def foo():
        d = {}
        for i in list(filter(lambda x: x.block_index > 0, rectangles)):
            if d.get(i.block_index):
                d[i.block_index] += 1
            else:
                d[i.block_index] = 1
        result = []
        for k, v in d.items():
            if v < 4:
                result.append(k)
        return result

    tmp = foo()
    for rect in rectangles:
        if (rect.block_index > 0 and rect.block_index in tmp) or (rect.block_index > 0 and rect.parent_index < 0):
            rect.block_index = 0

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
            tables.append(image[rect.top: rect.top + rect.height, rect.left: rect.left + rect.width])

    return tables

def parse_table(table: np.array):
    ret, thresh_value = cv2.threshold(table[:, :, 0], 75, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    obr_img = cv2.erode(thresh_value, kernel, iterations=1)
    dilated_value = cv2.GaussianBlur(obr_img, (3, 3), 0)
    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    # cluster[0] - delta <= center[0] <= cluster[0] + delta

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

        if len(same_clusters) > len(column_clusters.keys()) * (2 / 3): continue

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
    for column_idx, cl in enumerate(sorted(list(column_clusters.keys()))):
        cell = Cell(row=0, column=column_idx)
        cell.rectangles.update(list(set(column_clusters[cl]) & set(header_cluster)))
        table_object.header.append(cell)

    for row_idx, cl in enumerate(rows_centers[1:]):
        row = []
        for column_idx, r in enumerate(row_clusters[cl]):
            cell = Cell(row=row_idx, column=column_idx)
            cell.rectangles.update(r)
            row.append(cell)
        table_object.add_row(row)

    # for head_cell in table_object.header:
    #     for r in

    # for head_cell in table_object.header:
    #     for cell_rect in head_cell.rectangles:
    #         cv2.imwrite(f"tables/cell-{head_cell.column}-{cell_rect.index}.jpg", table[cell_rect.top: cell_rect.top + cell_rect.height, cell_rect.left: cell_rect.left + cell_rect.width])

    img1 = table.copy()
    for k in column_clusters:
        draw_rects(img1, column_clusters[k], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=5)



    img2 = table.copy()
    for k in row_clusters:
        draw_rects(img2, row_clusters[k], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                   thickness=5)
    # draw_rects(img2, row_clusters[rows_centers[0]], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    #            thickness=10)

    t_c = table.copy()
    draw_rects(t_c, rectangles)
    cv2.imwrite("tables/table-columns-rect.jpg", img1)
    cv2.imwrite("tables/table-rows-rect.jpg", img2)
    cv2.imwrite("tables/table-rect.jpg", t_c)

if __name__ == '__main__':
    test_pdf = 'pdf_examples/2 022_1_10_Счет СД Екб от 10.01.22 зак 2.pdf'
    # test_pdf = 'pdf_examples/Архитектурные.pdf'
    # test_pdf = 'pdf_examples/2 022_1_10_Счет_УИ_Екб_испр_от_11_01_2022_зак_5.pdf'
    # test_pdf = 'pdf_examples/Смета до пример.pdf'

    for page_idx, page_img in enumerate(pdf_to_images(test_pdf)):
        cv2.imwrite("pages/page-0.png", page_img)
        gray_image = page_img[:, :, 0]
        ret, thresh_value = cv2.threshold(gray_image, 75, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        obr_img = cv2.erode(thresh_value, kernel, iterations=1)
        dilated_value = cv2.GaussianBlur(obr_img, (3, 3), 0)
        contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ogr = round(max(page_img.shape[0], page_img.shape[1]) * 0.01)
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

        tables_images = get_tables(rectangles, page_img)

        for idx, table in enumerate(tables_images):
            cv2.imwrite(f"tables/table-{idx}.jpg", table)

        parse_table(tables_images[0])

        # coordinates = create_blocks(coordinates, delta)
        # coordinates = remove_trash(coordinates)
        # clustering(coordinates, delta)
        # print(len(coordinates))
        # print(len(list(filter(lambda x: x.parent_index == -1, coordinates))))
        draw_rects(page_img, rectangles, (0, 0, 255), 2)

        cv2.imwrite(f"pages/page-{page_idx}-rect.png", page_img)
        break