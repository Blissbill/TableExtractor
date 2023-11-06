import argparse
import csv
import os
import random
import time
from datetime import datetime
from enum import StrEnum

import requests
import tqdm


class MissingAnswer:
    DATE = "Дата"
    DOCUMENT_TYPE = "Тип документа"
    TABLE = "Таблица"
    CUSTOMER = "Покупатель"
    VENDOR = "Поставщик"
    NUMBER = "Номер"


parser = argparse.ArgumentParser()

parser.add_argument("--folder", default="pdf_examples", type=str)
parser.add_argument("--count", help="Number of pdf from the test folder. -1 for all files",
                    default=-1, type=int)
parser.add_argument("--random", help="Random Files. Works if not all files in the folder",
                    default=True, type=bool)
parser.add_argument("--result-folder", default="test_results", type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    files = os.listdir(args.folder)
    if args.random and args.count != -1:
        random.shuffle(files)
        files = files[:args.count]
    os.makedirs(args.result_folder, exist_ok=True)
    header = ["Файл", "Время обработки", "Количество ошибок", "Ошибки"]
    with open(os.path.join(args.result_folder, f"pdf test {time.time()}.csv"), "w", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, delimiter=';', fieldnames=header)
        csv_writer.writeheader()
        for file in tqdm.tqdm(files):
            file_path = os.path.join(args.folder, file)
            try:
                with open(file_path, "rb") as pdf_file:
                    start_time = datetime.now()
                    response = requests.post("http://127.0.0.1:8555/table_extractor/pdf", files={"pdf": pdf_file},
                                             data={"ocr": "easyocr"})
                    end_time = datetime.now()
                response.raise_for_status()
            except Exception as e:
                print(f"[{file}] {e}")
                continue
            data = response.json()
            exceptions = []
            if not data.get("date"):
                exceptions.append(MissingAnswer.DATE)
            if not data.get("document_type"):
                exceptions.append(MissingAnswer.DOCUMENT_TYPE)
            if not data.get("tables"):
                exceptions.append(MissingAnswer.TABLE)
            if not data.get("Покупатель"):
                exceptions.append(MissingAnswer.CUSTOMER)
            if not data.get("Поставщик"):
                exceptions.append(MissingAnswer.VENDOR)
            if not data.get("№"):
                exceptions.append(MissingAnswer.NUMBER)
            csv_writer.writerow({"Файл": file, "Время обработки": str(end_time - start_time),
                                 "Количество ошибок": str(len(exceptions)), "Ошибки": ".".join(exceptions)})
