# progress: Complete
# Run will succeed when pizzapy is the only project opened in PyCharm
# failed when the top level python folder is opened in PyCharm as Project
# https://stackoverflow.com/questions/51223647/how-can-i-elegantly-prevent-qthread-from-being-destroyed-or-garbage-collected


import sys
sys.path.append('..')


import time
from datetime import date
from functools import partial
from typing import Any, Generator, Dict


import pandas as pd
from PySide2.QtWidgets import QApplication, QCheckBox
from PySide2.QtCore import Qt, QThread, QCoreApplication, QSortFilterProxyModel



from shared_model.sql_model import cnx, cur, postgres_engine
from stock_core_browser.core_browser_view import GuruBrowserWin
from shared_model.st_data_model import getstocks

from dimsum.database.postgres import upsertquery
from dimsum.qt5.dataframemodel import DataFrameModel




class CustomSortFilterProxyModel(QSortFilterProxyModel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filterString = ''
        self.filterFunctions = {}

    def setFilterString(self, text):
        self.filterString = text.lower()
        self.invalidateFilter()

    def addFilterFunction(self, name, new_func):

        self.filterFunctions[name] = new_func
        self.invalidateFilter()

    def removeFilterFunction(self, name):

        if name in self.filterFunctions.keys():
            del self.filterFunctions[name]
            self.invalidateFilter()

    def filterAcceptsRow(self, row_num, parent):
        model = self.sourceModel()
        tests = [func(model.row(row_num), self.filterString)
                 for func in self.filterFunctions.values()]
        return False not in tests






class GuruBrowserDialog(GuruBrowserWin):
    def __init__(self) -> None:
        super().__init__()
        self.load_list_button.clicked.connect(self.load_table)
        self.load_symbols_button.clicked.connect(self.load_table)

        self.b_list_zacks.clicked.connect(self.load_table)
        self.b_le_zacks.clicked.connect(self.load_table)

    def load_table(self) -> None:
        self.clear()
        sender = self.sender().accessibleName()
        if sender == 'load_list_button':
            tablename = 'usstock_g'
            stockstr = self.stock_list_combobox.currentText()
            stocklist = getstocks(stockstr)
            stockliststr = str(tuple(stocklist))
        elif sender == 'load_symbols_button':
            tablename = 'usstock_g'
            stockstr = self.symbols_lineedit.text()
            stocklist = stockstr.split()
            stockliststr = str(stocklist).replace('[','(').replace(']',')')  # for single tuple

        elif sender == 'b_list_zacks':
            tablename = 'usstock_z'
            stockstr = self.stock_list_combobox.currentText()
            stocklist = getstocks(stockstr)
            stockliststr = str(tuple(stocklist))
        elif sender == 'b_le_zacks':
            tablename = 'usstock_z'
            stockstr = self.symbols_lineedit.text()
            stocklist = stockstr.split()
            stockliststr = str(stocklist).replace('[', '(').replace(']', ')')  # for single tuple

        q_clause = '' if not stocklist else ' WHERE symbol IN ' + stockliststr  # prevent empty LineEdit
        q = 'SELECT * FROM ' + tablename + q_clause
        df = pd.read_sql(sql=q, con=postgres_engine)
        model = DataFrameModel(df)
        self.sort_filter_model = CustomSortFilterProxyModel()
        self.sort_filter_model.setSourceModel(model)
        self.pandas_tableview.setModel(self.sort_filter_model)
        self.pandas_tableview.model().addFilterFunction(
            'allcolumns',
            lambda r, s: (True in [s in str(col).lower()
                                   for col in r]))

        checkboxes = [QCheckBox(x) for x in df.columns]
        for count, checkbox in enumerate(checkboxes):
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(partial(self.display_column, index=count))
            self.dockwinbox.addWidget(checkbox)


    def display_column(self, state, index=None) -> None:
        #print(self, checkbox, state)
        if state == Qt.Checked:
            self.pandas_tableview.setColumnHidden(index, False)
        else:
            self.pandas_tableview.setColumnHidden(index, True)


def main() -> None:
    app = QApplication(sys.argv)
    w = GuruBrowserDialog()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
