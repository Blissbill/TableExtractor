from typing import List, Set

import numpy as np
from pydantic import BaseModel


class Rectangle(BaseModel):
    index: int
    left: int
    top: int
    width: int
    height: int
    parent_index: int = -1
    text: str = ""

    def __str__(self):
        return f"#{self.index} [{self.left}, {self.top}; {self.width}, {self.height}]"

    def __hash__(self):
        return self.__str__().__hash__()


class Cell(BaseModel):
    column: int
    row: int
    text: str = ""


class Table:
    def __init__(self):
        self.header: List[Cell] = []
        self.__data: "np.array[np.array[Cell]]" = []

    def add_row(self, row: List[Cell]):
        if len(row) != len(self.header):
            raise Exception("Added row and header have different sizes")
        self.__data.append(np.array(row))

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.__data[item[0]][item[1]]
        else:
            return self.__data[item]

    def to_dict(self):
        jn = {"header": [], "rows": []}
        for h in self.header:
            jn["header"].append(h.text)
        for r in self.__data:
            row = []
            for d in r:
                row.append(d.text)
            jn["rows"].append(row)
        return jn
