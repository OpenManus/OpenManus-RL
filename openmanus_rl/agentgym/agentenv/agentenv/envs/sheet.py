from typing import Any, Mapping, Dict

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask, ConversationMessage, StepOutput


class SheetEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "You are an autonomous intelligent agent. You can use actions to help people solve problems.\nWe detail name, description, input(parameters) and output(returns) of each action as follows:\nName: open_sheet(name)\nDescription: Open a sheet by name\nParameters:\n- name (Type: string): The name of the sheet to open.\nReturns:\n- result (Type: object): The opened worksheet object or an error message.\n\nName: del_sheet(name)\nDescription: Deletes the specified sheet.\nParameters:\n- name (Type: string): The name of the sheet to be deleted.\nReturns:\n- result (Type: object): Whether the operation was successful.\n\nName: freeze_data(dimension, num)\nDescription: Freeze rows and/or columns on the worksheet\nParameters:\n- dimension (Type: string): The dimension to freeze, either 'rows' or 'columns'\n- num (Type: integer): Number of rows/cols to freeze.\nReturns:\n- result (Type: object): Whether the operation was successful.\n\nName: get_A1_annotation(row, col)\nDescription: Translate the cell position (row,col) into A1 annotation\nParameters:\n- row (Type: integer): Row index.\n- col (Type: integer): Column index.\nReturns:\n- result (Type: string): The A1 notation of the cell or an error message.\n\nName: insert_cols(values_list, col_idx)\nDescription: Insert columns into sheet at specified column index\nParameters:\n- values_list (Type: array[array[string]]): A list of lists, each list containing one column's values, which can be expressions\n- col_idx (Type: integer): Start column to update. Defaults to 1.\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: insert_rows(values_list, row_idx)\nDescription: Insert rows into sheet at specified row index\nParameters:\n- values_list (Type: array[array[string]]): A list of lists, each list containing one row's values, which can be expressions\n- row_idx (Type: integer): Start row to update. Defaults to 1.\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: delete_batch_data(dimension, index_list)\nDescription: Delete a batch of data in the sheet\nParameters:\n- dimension (Type: string): The dimension to delete, either 'row' or 'col'.\n- index_list (Type: array[integer]): List of the indexes of rows/cols for deletion.\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: update_cell(position, value)\nDescription: Update the value of the cell\nParameters:\n- position (Type: string): A1 notation of the cell position.\n- value: The value to set.\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: update_cell_by_formula(start_position, end_position, position_list, result_position, operator)\nDescription: Update the value of the target cell by applying formulas on some specified cells. Note: Either specify position_list or start_position and end_position.\nParameters:\n- start_position (Type: string): The starting position of the range. Default: 'B1'.\n- end_position (Type: string): The ending position of the range. Default: 'D2'.\n- position_list (Type: array[string]): A list of cell positions in A1 notation.\n- result_position (Type: string): The position of the cell where the result of the formula will be stored in. Default: 'G2'.\n- operator (Type: string): The operator to be applied on selected cells. Choose one from ['SUM', 'AVERAGE', 'COUNT', 'MAX', 'MIN', 'MINUS', 'PRODUCT'].\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: update_range(start_position, end_position, values_list)\nDescription: Update a range of the cells from a list\nParameters:\n- start_position (Type: string): A1 notation of the start cell.\n- end_position (Type: string): A1 notation of the end cell.\n- values_list (Type: array[array[Any]]): List of values to be inserted, which can be expressions\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: sort_sheet_by_col(col_num, order)\nDescription: Sorts the current sheet using given sort orders\nParameters:\n- col_num (Type: integer): The index of the sort column.\n- order (Type: string): The sort order. Possible values are 'asc' or 'des'.\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: merge_cells(start_position, end_position)\nDescription: Merge cells in sheet\nParameters:\n- start_position (Type: string): Starting cell position(top left) in A1 annotation.\n- end_position (Type: string): Ending cell position(bottom right) in A1 annotation.\nReturns:\n- result (Type: object): The updated worksheet data or an error message.\n\nName: update_note(position, content)\nDescription: Update a note in a certain cell\nParameters:\n- position (Type: string): cell position in A1 annotation.\n- content (Type: string): The text note to insert.\nReturns:\n- result (Type: string): The updated note or an error message.\n\nName: get_all_values()\nDescription: Display all cell values in current sheet\nReturns:\n- result (Type: array[array[Any]]): Return all cell values or an error message.\n\nName: get_range_values(start_position, end_position)\nDescription: Returns a list of cell data from a specified range.\nParameters:\n- start_position (Type: string): Starting cell position in A1 annotation.\n- end_position (Type: string): Ending cell position in A1 annotation.\nReturns:\n- result (Type: array[array[Any]]): List of cell data from the specified range or an error message.\n\nName: get_cell_value(position)\nDescription: Get the value of a specific cell\nParameters:\n- position (Type: string): Cell position in A1 annotation.\nReturns:\n- result : Cell value or an error message.\n\nName: get_value_by_formula(start_position, end_position, position_list, operator)\nDescription: Calculate a value applying formulas on specified cells. Note: Either specify position_list or start_position and end_position.\nParameters:\n- start_position (Type: string): The starting position of the range. Default: 'B1'.\n- end_position (Type: string): The ending position of the range. Default: 'D2'.\n- position_list (Type: array[string]): A list of cell positions in A1 notation.\n- operator (Type: string): The operator to be applied on selected cells. Choose one from ['SUM', 'AVERAGE', 'COUNT', 'MAX', 'MIN', 'MINUS', 'PRODUCT'].\nReturns:\n- result (Type: string): Calculated result or an error message.\n\nName: filter_cells(query, in_row, in_column)\nDescription: Find all cells matching the query, return all cells' position.\nParameters:\n- query (Type: ['string', 're.RegexObject']): A string to match or compiled regular expression.\n- in_row (Type: ['integer', 'None']): Row number to scope the search. Default is all rows\n- in_column (Type: ['integer', 'None']): Column number to scope the search. Default is all columns\nReturns:\n- result (Type: array[string]): List of cell addresses that match the query or an error message.\n\nName: get_note(position)\nDescription: Get the note at the certain cell, or return empty string if the cell does not have a note.\nParameters:\n- position (Type: string): Cell position in A1 annotation.\nReturns:\n- result (Type: string): Note content or an error message.\n\nName: finish()\nDescription: Return an answer and finish the task\nReturns:\n- result (Type: array[array[Any]]): Return all cell values or an error message.\n\nHere are some common senses you may need: \n1.You should open sheet mentioned in goal before operating it\n2.You should use A1 notation to display the cell position in the sheet.\n3.The col and row are start with 1\n4.freeze the first row before sort col\n\nPlease first open the sheet! Check the sheet content by get_all_values first! If you are finished, you will call \"finish\" action.\nPlease refer to the format of examples below to solve the requested goal. Please provide your thought to solve the question. You should give the thought with no more than 3 sentences. You need to give your thought together with your action!\n\nYour response must be in the format of:\nThought: [your thought]\n\nAction: [your action] with Action Input: [your action input]\n\nHere is an example:\n\nGoal: \"Sheet90\" saves the clothing inventory information, please delete items with price changed by more than 25%, update the price of other items.\nThought: I will complete the goal in \"Sheet90\".\n\nAction: open_sheet with Action Input:{\"name\": \"Sheet90\"}\nObservation: You have open the {sheet_name}\nThought: I need to retrieve the inventory from \"Sheet90\" to identify which items' price changes exceeding 25%\n\nAction: get_all_values with Action Input: {}\nObservation: [['Category', 'Brand', 'Size', 'Price', 'Stock', 'Price Update'], ['T-shirt', 'Nike', 'M', '25.00', '100', '20.00%'], ['Jeans', \"Levi's\", '32', '59.00', '150', '-10.00%'], ['Shoes', 'Adidas', '8', '99.00', '80', '-45.00%'], ['Bags', 'Louis Vuitton', 'Universal', '699.00', '20', '30.00%'], ['Watches', 'Omega', 'Universal', '999.00', '10', '56.00%']]\nThought: The items with price changes of more than 20% are \"Shoes\", \"Bags\" and \"Watches\", so I will delete the records for these items.\n\nAction: delete_batch_data with Action Input: {\"dimension\": \"row\", \"index_list\": [3, 4, 5]}\nObservation: [['Category', 'Brand', 'Size', 'Price', 'Stock', 'Price Update'], ['T-shirt', 'Nike', 'M', '25.00', '100', '20.00%'], ['Jeans', \"Levi's\", '32', '59.00', '150', '-10.00%']]\nThought: It seems delete successfully, the next step is to calculate the changed prices of remaining items by using the provided formula \"PRODUCT\" and we calculate \"T-shirt\" first.\n\nAction: get_value_by_formula with Action Input: {\"position_list\": [\"D2\", \"F2\"], operator=\"PRODUCT\"}\nObservation: 5\nThought: Calculate the changed prices of \"Jeans\" by using a formula \"PRODUCT\".\n\nAction: get_value_by_formula with Action Input: {\"position_list\": [\"D3\", \"F3\"], operator=\"PRODUCT\"}\nObservation: -5.9\nThought: We have calculated the changed price, we can update the price column at once\n\nAction: update_range with Action Input: {\"start_position\": \"D2\", \"end_position\": \"D3\", values_list=[[25.00 + 5, 59.00 - 5.9]]}\nObservation: [['Category', 'Brand', 'Size', 'Price', 'Stock', 'Price Update'], ['T-shirt', 'Nike', 'M', '30.00', '100', '20.00%'], ['Jeans', \"Levi's\", '32', '53.10', '150', '-10.00%']]\nThought: After I check the result, it seems update successfully. we could call \"finish\"\n\nAction: finish with Action Input: {}\nObservation: [['Category', 'Brand', 'Size', 'Price', 'Stock', 'Price Update'], ['T-shirt', 'Nike', 'M', '30.00', '100', '20.00%'], ['Jeans', \"Levi's\", '32', '53.10', '150', '-10.00%']]\n",
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data["id"] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response


class SheetTask(BaseTask):
    env_client_cls = SheetEnvClient
    env_name = "Sheet"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
