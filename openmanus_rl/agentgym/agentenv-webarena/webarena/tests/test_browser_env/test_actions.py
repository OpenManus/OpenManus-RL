import numpy as np

from browser_env import *


def test_is_equivalent() -> None:
    for action_type in ActionTypes.__members__.values():
        action_a = create_random_action()
        action_b = create_random_action()
        if action_a["action_type"] != action_b["action_type"]:
            assert not is_equivalent(action_a, action_b)
        action_a["action_type"] = action_type
        action_b["action_type"] = action_type
        match action_type:
            case ActionTypes.MOUSE_CLICK | ActionTypes.MOUSE_HOVER:
                if not np.allclose(action_a["coords"], action_b["coords"]):
                    assert not is_equivalent(action_a, action_b)
                    action_a["coords"] = action_b["coords"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.KEYBOARD_TYPE:
                if action_a["text"] != action_b["text"]:
                    assert not is_equivalent(action_a, action_b)
                    action_a["text"] = action_b["text"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.CLICK | ActionTypes.HOVER | ActionTypes.TYPE:
                if action_a["element_id"] and action_b["element_id"]:
                    if action_a["element_id"] != action_b["element_id"]:
                        assert not is_equivalent(action_a, action_b)
                        action_a["element_id"] = action_b["element_id"]
                    assert is_equivalent(action_a, action_b)
                elif action_a["element_id"] and action_b["element_id"]:
                    if action_a["element_role"] != action_b["element_role"]:
                        assert not is_equivalent(action_a, action_b)
                        action_a["element_role"] = action_b["element_role"]
                    if action_a["element_name"] != action_b["element_name"]:
                        assert not is_equivalent(action_a, action_b)
                        action_a["element_name"] = action_b["element_name"]
                    assert is_equivalent(action_a, action_b)
                elif action_a["pw_code"] and action_b["pw_code"]:
                    if action_a["pw_code"] != action_b["pw_code"]:
                        assert not is_equivalent(action_a, action_b)
                        action_a["pw_code"] = action_b["pw_code"]
                    assert is_equivalent(action_a, action_b)
                else:
                    action_a["element_id"] = action_b["element_id"]
                    assert is_equivalent(action_a, action_b)
            case ActionTypes.GOTO_URL:
                if action_a["url"] != action_b["url"]:
                    assert not is_equivalent(action_a, action_b)
                    action_a["url"] = action_b["url"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.PAGE_FOCUS:
                if action_a["page_number"] != action_b["page_number"]:
                    assert not is_equivalent(action_a, action_b)
                    action_a["page_number"] = action_b["page_number"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.SCROLL:
                da = "up" if "up" in action_a["direction"] else "down"
                db = "up" if "up" in action_b["direction"] else "down"
                if da != db:
                    assert not is_equivalent(action_a, action_b)
                    action_a["direction"] = action_b["direction"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.KEY_PRESS:
                if action_a["key_comb"] != action_b["key_comb"]:
                    assert not is_equivalent(action_a, action_b)
                    action_a["key_comb"] = action_b["key_comb"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.CHECK | ActionTypes.SELECT_OPTION:
                if action_a["pw_code"] != action_b["pw_code"]:
                    assert not is_equivalent(action_a, action_b)
                    action_a["pw_code"] = action_b["pw_code"]
                assert is_equivalent(action_a, action_b)
            case ActionTypes.STOP:
                if action_a["answer"] != action_b["answer"]:
                    assert not is_equivalent(action_a, action_b)
                    action_a["answer"] = action_b["answer"]
                assert is_equivalent(action_a, action_b)
            case _:
                assert is_equivalent(action_a, action_b)


def test_action2create_function() -> None:
    for _ in range(1000):
        action = create_random_action()
        create_function = action2create_function(action)
        assert is_equivalent(action, eval(create_function))
