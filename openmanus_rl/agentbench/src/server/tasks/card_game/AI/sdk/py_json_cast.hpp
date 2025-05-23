#pragma once

#ifndef PY_JSON_CAST_HPP_INCLUDED
#define PY_JSON_CAST_HPP_INCLUDED

#include <pybind11/pybind11.h>

#include "jsoncpp/json/json.h"

namespace pybind11::detail {
template <>
class type_caster<Json::Value> {
   public:
    PYBIND11_TYPE_CASTER(Json::Value, _("Json"));

    bool load(handle src, bool) try {
        value = handle_to_json(src);
        return true;
    } catch (type_error) {
        return false;
    }

    static handle cast(Json::Value src, return_value_policy, handle) {
        return json_to_handle(src).release();
    }

   private:
    static Json::Value handle_to_json(const handle &hdl) {
        if (hdl.ptr() == nullptr || hdl.is_none()) return Json::nullValue;
        if (isinstance<bool_>(hdl)) return hdl.cast<bool>();
        if (isinstance<int_>(hdl)) return hdl.cast<Json::LargestInt>();
        if (isinstance<float_>(hdl)) return hdl.cast<double>();
        if (isinstance<str>(hdl)) return hdl.cast<std::string>();
        if (isinstance<tuple>(hdl) || isinstance<list>(hdl) ||
            isinstance<set>(hdl)) {
            Json::Value ret = Json::arrayValue;
            for (const handle &h : hdl) ret.append(handle_to_json(h));
            return ret;
        }
        if (isinstance<dict>(hdl)) {
            Json::Value ret = Json::objectValue;
            for (const handle &key : hdl)
                ret[str(key)] = handle_to_json(hdl[key]);
            return ret;
        }
        throw type_error("Bad cast from Python to C++: " +
                         repr(hdl).cast<std::string>());
    }
    static object json_to_handle(const Json::Value &json) {
        if (json.isNull()) return none();
        if (json.isBool()) return bool_(json.asBool());
        if (json.isIntegral()) return int_(json.asLargestInt());
        if (json.isNumeric()) return float_(json.asDouble());
        if (json.isString()) return str(json.asCString());
        if (json.isArray()) {
            list ret;
            for (const Json::Value &j : json) ret.append(json_to_handle(j));
            return ret;
        }
        if (json.isObject()) {
            dict ret;
            for (Json::ValueConstIterator iter = json.begin();
                 iter != json.end(); ++iter)
                ret[str(iter.key().asCString())] = json_to_handle(*iter);
            return ret;
        }
        throw type_error("Bad cast from C++ to Python: " + json.asString());
    }
};
}  // namespace pybind11::detail

#endif  // PY_JSON_CAST_HPP_INCLUDED