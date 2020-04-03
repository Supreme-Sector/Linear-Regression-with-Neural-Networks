import xlrd
import numpy as np

def get_data():
    wb = xlrd.open_workbook("data.xlsx")
    sheet = wb.sheet_by_index(0)
    rows = sheet.nrows

    data = []
    for i in range(1, rows):
        x = np.array([[sheet.cell_value(i, 0)]])
        y = np.array([[sheet.cell_value(i, 1)]])
        data.append((x, y))

    return data