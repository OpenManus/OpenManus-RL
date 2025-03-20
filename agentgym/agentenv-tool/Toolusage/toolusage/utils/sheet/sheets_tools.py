import os
import time
import re
from datetime import datetime
import logging
import gspread
from gspread.utils import Dimension
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_PATH = os.getenv("PROJECT_PATH")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


AUTHORIZATION = {
    "API_JSON": f"{PROJECT_PATH}/toolusage/utils/sheet/credential.json",
    "REFERENCE_SHEET_ID": "17MVNCONh-6Pw3Met2O31WhUHlgOcQz8TPJKikVpL8IY",
    "YOUR_EMAIL": os.getenv("SHEET_EMAIL")
}


def validate_A1_annotation(input_str):
    pattern = r"^[A-Z]{1}[0-9]+$"
    match = re.match(pattern, input_str)
    if match:
        return True
    else:
        return False


def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if len(lst) > 0 else None
    return shape


def check_2d_list(input_list):
    if isinstance(input_list, list) and all(isinstance(sub_list, list) for sub_list in input_list):
        return True
    else:
        error = "The values_list needs to be a 2D list, please check your parameter"
        return False, repr(error)


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def convert_range_to_float(data):
    converted_data = []
    for row in data:
        converted_row = [float(value) if is_numeric(value) else value for value in row]
        converted_data.append(converted_row)
    return converted_data

def convert_value_to_float(value):
    if is_numeric(value):
        return float(value)
    else:
        return value

def log_path(func):
    def wrapper(self, *args, **kwargs):
        if func.__name__ == "open_sheet" or func.__name__ == "delete_sheet":
            pass
        else:
            if not hasattr(self, 'worksheet'):
                error = "Please open a sheet before operating it"
                return error
            # print(f"Now Operate Sheet: {self.worksheet}")
        if "action_path" in kwargs.keys():
            action_path = kwargs["action_path"]
            kwargs.pop("action_path")
            success, result = func(self, *args, **kwargs)

            # convert value in kwargs to string
            for key, value in kwargs.items():
                kwargs[key] = str(value)

            if success:
                action_path.append({
                    "Action": func.__name__,
                    "Action Input": str(kwargs),
                    "Observation": "Calling " + func.__name__ + " with " + str(kwargs) + " successfully." + str(result),
                })
                return result
            else:
                action_path.append({
                    "Action": func.__name__,
                    "Action Input": str(kwargs),
                    "Observation": "Calling " + func.__name__ + " with " + str(kwargs) + " failed." + result,
                })
                return result
        else:
            return func(self, *args, **kwargs)

    return wrapper


class sheet_toolkits:
    def __init__(self):
        # Authenticate using API credentials
        api_json = AUTHORIZATION["API_JSON"]
        self.client = gspread.service_account(api_json)

    def init_sheets(self):
        print(">>>>>>>>>Init<<<<<<<<<<")

        # Check if the API_JSON file exists
        if not os.path.exists(AUTHORIZATION["API_JSON"]):
            raise Exception("Please get the API_JSON file from Google Cloud Platform and put it in `./utils/sheet/` folder") 
        while True:
            try:
                gc = gspread.service_account(filename=AUTHORIZATION["API_JSON"])  # Please use your client.

                ss_copy = gc.copy(file_id=AUTHORIZATION["REFERENCE_SHEET_ID"])
                res = ss_copy.share(email_address=AUTHORIZATION["YOUR_EMAIL"], perm_type="user", role="writer", notify=False)
                permissionId = res.json()["id"]
                ss_copy.transfer_ownership(permissionId)
                self.spreadsheet_id = ss_copy.id
                self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                print("Created Success")
                print(f"SpreadSheet_id:{self.spreadsheet_id}")
                break
            except Exception as error:
                print(error)
                time.sleep(10)

    def for_test(self):
        # Get the unique identifier of the spreadsheet
        self.spreadsheet_id = "1RN2jyxoowTCS04Ct4v8Q-_OrOaBgViz3XjWgckuxl5k"
        self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)

    @log_path
    def delete_sheets(self):
        self.client.del_spreadsheet(self.spreadsheet_id)
        print("deleted sheets")

    @log_path
    def open_sheet(self, name):
        try:
            self.worksheet = self.spreadsheet.worksheet(name)
        except Exception as error:
            return False, repr(error)
        return True, self.worksheet

    @log_path
    def del_sheet(self, name):
        try:
            self.spreadsheet.del_worksheet(name)
        except Exception as error:
            return False, repr(error)
        return True, f"You have deleted sheet {name} successfully"

    ###### UTILS #######
    # Freeze rows and/or columns on the worksheet;
    # dimension((rows/columns) - freeze row/col);
    # num((int) - Number of rows/cols to freeze.)
    @log_path
    def freeze_data(self, dimension, num):
        try:
            num = int(num)
            if dimension == "rows":
                self.worksheet.freeze(rows=num, cols=None)
            elif dimension == "columns":
                self.worksheet.freeze(rows=None, cols=num)
            else:
                error = "Error: the dimension should be one of the rows/columns"
                return False, repr(error)
        except Exception as error:
            return False, repr(error)
        return True, f"You have successfully freeze the {dimension} {num}"

    # Translate the (row,col) into A1 annotation to get the cell position in A1 annotation;
    # row(int); col(int)
    @log_path
    def get_A1_annotation(self, row, col):
        try:
            result = self.worksheet.cell(int(row), int(col)).address
        except Exception as error:
            return False, repr(error)
        return True, result

    ###### ADD #######
    # insert cols into sheet in col "col_idx";
    # values_list((list(list)) - a list of lists, with the lists each containing one col’s values);
    # col_idx((int) – Start col to update. Defaults to 1.)
    @log_path
    def insert_cols(self, values_list, col_idx):
        try:
            if check_2d_list(values_list):
                values_list = convert_range_to_float(values_list)
            else:
                return check_2d_list(values_list)
            col_num = len(self.get_all_values()[1][0])
            if int(col_idx) == col_num + 1:
                self.worksheet.insert_cols(values_list, col_idx, value_input_option='RAW', inherit_from_before=True)
            else:
                self.worksheet.insert_cols(values_list, col_idx, value_input_option='RAW',
                                           inherit_from_before=False)
        except Exception as error:
            return False, repr(error)
        return True, self.get_all_values()[1]

    # insert raws into sheet in raw "raw_idx";
    # values_list((list(list)) - a list of lists, with the lists each containing one row’s values);
    # row_idx((int) – Start row to update. Defaults to 1.)
    @log_path
    def insert_rows(self, values_list, row_idx):
        try:
            if check_2d_list(values_list):
                values_list = convert_range_to_float(values_list)
            else:
                return check_2d_list(values_list)
            raw_num = len(self.get_all_values()[1])
            if int(row_idx) == raw_num + 1:
                self.worksheet.insert_rows(values_list, row_idx, value_input_option='RAW', inherit_from_before=True)
            else:
                self.worksheet.insert_rows(values_list, row_idx, value_input_option='RAW',
                                           inherit_from_before=False)
        except Exception as error:
            return False, repr(error)
        return True, self.get_all_values()[1]


    ###### DELETE #######
    # delete range of data; dimension("row"/"col") - delete dimension; start_index(int) - Index of a first row/col /
    # for deletion; end_index(int) - Index of a last row/col for deletion. if None, only delete a single row/col at /
    # start_index.
    # def delete_range(self, dimension, start_index, end_index=None):
    #     try:
    #         if dimension == "row":
    #             self.worksheet.delete_dimension(Dimension.rows, start_index, end_index)
    #         elif dimension == "col":
    #             self.worksheet.delete_dimension(Dimension.cols, start_index, end_index)
    #         else:
    #             error = "Wrong dimension, please choose from row/col"
    #             return False, repr(error)
    #     except Exception as error:
    #         return False, repr(error)
    #     return True, self.get_all_values()[1]

    # delete range of data; dimension("row"/"col") - delete dimension; start_index(list(int)) - list of the index /
    # of rows/cols for deletion
    @log_path
    def delete_batch_data(self, dimension, index_list):
        try:
            sorted_list = sorted(index_list, key=lambda x: int(x), reverse=True)
            if dimension == "row":
                rows = len(self.get_all_values()[1])
                if len(index_list) >= rows - 1:
                    error = "Warning: You cannot delete whole sheet at once, please try other functions"
                    return False, repr(error)
                for i in sorted_list:
                    time.sleep(0.2)
                    self.worksheet.delete_dimension(Dimension.rows, i)
            elif dimension == "col":
                cols = len(self.get_all_values()[1][0])
                if len(index_list) == cols:
                    error = "Warning: You cannot delete whole sheet at once, please try other functions"
                    return False, repr(error)
                for i in sorted_list:
                    time.sleep(0.2)
                    self.worksheet.delete_dimension(Dimension.cols, i)
            else:
                error = "Wrong dimension, please choose from row/col"
                return False, repr(error)
        except Exception as error:
            return False, repr(error)
        return True, self.get_all_values()[1]

    ###### MODIFY #######
    # Update the value of the cell by setting value
    @log_path
    def update_cell(self, position, value):
        try:
            value = convert_value_to_float(value)
            self.worksheet.update_acell(position, value)
        except Exception as error:
            return False, repr(error)
        return True, self.get_all_values()[1]

    # Update the value of the target cell by using formulas on other cells
    # here has two type representing ranges: [start_position:end_position](str,str) or position_list(list(str)), \
    # please choose one of them according to your needs, and note that all positions should be A1 annotation
    @log_path
    def update_cell_by_formula(self, start_position="B1", end_position="D2", position_list=None, result_position="G2",
                               operator=""):
        try:
            if operator not in ["SUM", "AVERAGE", "COUNT", "MAX", "MIN", "MINUS", "PRODUCT"]:
                error = """the operator should be one of the ["SUM", "AVERAGE", "COUNT", "MAX", "MIN", "MINUS", "PRODUCT"]"""
                return False, repr(error)
            if position_list is None:
                if validate_A1_annotation(start_position) and validate_A1_annotation(end_position) and \
                        validate_A1_annotation(result_position):
                    overall_formula = f"={operator}({start_position}:{end_position})"
                    self.worksheet.update(result_position, overall_formula, raw=False)
                else:
                    error = ("In the sheet formula, you should use A1 notation to display all positions in the "
                             "sheet and you can use get_A1_annotation(row, col) to get its A1 address.")
                    return False, repr(error)
            else:
                if all(validate_A1_annotation(position) for position in position_list) and \
                        validate_A1_annotation(result_position):
                    positions = ",".join(position_list)
                    overall_formula = f"={operator}({positions})"
                    self.worksheet.update(result_position, overall_formula, raw=False)
                else:
                    error = ("In the sheet formula, you should use A1 notation to display all positions in the "
                             "sheet and you can use get_A1_annotation(row, col) to get its A1 address.")
                    return False, repr(error)

            return True, self.get_all_values()[1]
        except Exception as error:
            return False, repr(error)

    # update a range of sheet from a list, should notice the shape of values_list
    @log_path
    def update_range(self, start_position, end_position, values_list):
        try:
            # if get_shape(self.get_range_values(start_position, end_position)) == get_shape(values_list):
            #     print()
            #     pass
            # else:
            #     error = "The values_list to be inserted is different in shape from the selected range."
            #     return False, repr(error)
            max_rows = len(self.get_all_values()[1])
            max_cols = len(self.get_all_values()[1][0])

            last_position = self.worksheet.cell(max_rows, max_cols).address
            if start_position == "A1" and end_position == last_position:
                error = "Warning: You cannot update whole sheet at once, please try other functions"
                return False, repr(error)
            values_list = convert_range_to_float(values_list)

            self.worksheet.update(f"{start_position}:{end_position}", values_list)
        except Exception as error:
            return False, repr(error)
        return True, self.get_all_values()[1]

    # Sorts worksheet using given sort orders;
    # col_num((int) - the index of the sort column);
    # order((‘asc’ or ‘des’) - the order);
    @log_path
    def sort_sheet_by_col(self, col_num, order):
        try:
            self.worksheet.sort((int(col_num), order))
        except Exception as error:
            return False, repr(error)
        return True, self.get_all_values()[1]

    # Merge cells.
    # start_position(str) - Starting cell position(top left) in A1 annotation
    # end_position(str) - Ending cell position(bottom right) in A1 annotation
    @log_path
    def merge_cells(self, start_position, end_position):
        try:
            self.worksheet.merge_cells(f"{start_position}:{end_position}", merge_type='MERGE_ALL')
        except Exception as error:
            return False, repr(error)
        # time.sleep(1)
        return True, self.get_all_values()[1]

    #  Update a note in a certain cell;
    #  position(str) - Cell position in A1 annotation
    #  content(str) - The text note to insert
    @log_path
    def update_note(self, position, content):
        try:
            self.worksheet.update_note(position, content)
        except Exception as error:
            return False, repr(error)
        result = "Updated! the {}'s note is {}".format(position, self.get_note(position)[1])
        return True, result

    ###### CHECK #######
    # display all cell values in worksheet
    # rewrite
    @log_path
    def get_all_values(self):
        try:
            max_rows = len(self.worksheet.get_all_values())
            max_cols = len(self.worksheet.get_all_values()[0])
            last_position = self.worksheet.cell(max_rows, max_cols).address
            result = self.worksheet.get_values(f"A1:{last_position}", combine_merged_cells=True)
        except Exception as error:
            return False, repr(error)
        return True, result

    # Returns a list of cell data from a specified range.
    @log_path
    def get_range_values(self, start_position, end_position):
        try:
            result = self.worksheet.get_values(f"{start_position}:{end_position}", combine_merged_cells=True)
        except Exception as error:
            return False, repr(error)
        return True, result

    # get cell value
    # position(str) - Cell position in A1 annotation
    @log_path
    def get_cell_value(self, position):
        try:
            result = self.worksheet.acell(position).value
        except Exception as error:
            return False, repr(error)
        return True, result

    # calculate by using formulas on cells;
    # here has two type representing ranges: [start_position:end_position](str,str) or position_list(list(str)), \
    # please choose one of them according to your needs, and note that all positions should present by A1 annotation
    @log_path
    def get_value_by_formula(self, start_position='B1', end_position='D2', position_list=None, operator=""):
        max_cols = len(self.worksheet.get_all_values()[0])
        self.insert_cols([[1]], max_cols+1)
        templete_cell = self.worksheet.cell(1, max_cols+1).address  # used for save result temporarily
        
        try:
            if operator not in ["SUM", "AVERAGE", "COUNT", "MAX", "MIN", "MINUS", "PRODUCT"]:
                self.worksheet.delete_columns(max_cols+1)
                error = """the operator should be one of the ["SUM", "AVERAGE", "COUNT", "MAX", "MIN", "MINUS", "PRODUCT"]"""
                return False, repr(error)
            if position_list is None:
                if validate_A1_annotation(start_position) and validate_A1_annotation(end_position):
                    overall_formula = f"={operator}({start_position}:{end_position})"
                    self.worksheet.update(templete_cell, overall_formula, raw=False)
                    result = self.worksheet.acell(templete_cell).value
                    self.worksheet.delete_columns(max_cols + 1)
                else:
                    error = "In the sheet formula, you should use A1 notation to display the start and end cell address in the " \
                            "sheet and you can use get_A1_annotation(row, col) to get its A1 address."
                    return False, repr(error)
            else:
                if all(validate_A1_annotation(position) for position in position_list):
                    positions = ",".join(position_list)
                    overall_formula = f"={operator}({positions})"
                    self.worksheet.update(templete_cell, overall_formula, raw=False)
                    result = self.worksheet.acell(templete_cell).value
                    self.worksheet.delete_columns(max_cols + 1)
                else:
                    error = ("In the sheet formula, you should use A1 notation to display all positions in the "
                             "sheet and you can use get_A1_annotation(row, col) to get its A1 address.")
                    return False, repr(error)

            return True, result
        except Exception as error:
            return False, repr(error)

    # find all cells matching the query, return all cells' position(list);
    # query(str, re.RegexObject) - A literal string to match or compiled regular expression.
    # in_row(int, None) - row number to scope the search
    # in_column(int, None) - col number to scope the search
    @log_path
    def filter_cells(self, query, in_row=None, in_column=None):
        try:
            in_row = int(in_row) if in_row is not None else None
            in_column = int(in_column) if in_column is not None else None
            position = []
            if isinstance(query, str):
                query = re.compile(query)
            results = self.worksheet.findall(query, in_row, in_column, case_sensitive=True)
            for cell in results:
                position.append(cell.address)
        except Exception as error:
            return False, repr(error)
        return True, position

    # Get the content of the note located at cell, or the empty string if the cell does not have a note;
    # position(str) - Cell position in A1 annotation
    @log_path
    def get_note(self, position):
        try:
            result = self.worksheet.get_note(position)
        except Exception as error:
            return False, repr(error)
        return True, result

    @log_path
    def finish(self):
        return True, self.get_all_values()[1]


if __name__ == "__main__":
    tool = sheet_toolkits()
    tool.for_test()
    _, worksheet = tool.open_sheet("Sheet1_1")
    print(tool.sort_sheet_by_col(7, "des"))
