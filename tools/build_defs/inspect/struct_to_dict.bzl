# Copyright 2021 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""struct_to_dict() tries to convert struct-like objects to dicts "recursively".

Useful to dump arbitrary provider data. Objects below the
specified depth are copied literally.
"""

load("@rules_java//java/common:java_info.bzl", "JavaInfo")

def struct_to_dict(x, depth = 5):
    root = {}
    queue = [(root, x)]
    for i in range(depth):
        nextlevel = [] if i < depth - 1 else None
        for dest, obj in queue:
            if _is_depset(obj):
                obj = obj.to_list()
            if _is_list(obj):
                for item in list(obj):
                    converted = _convert_one(item, nextlevel)
                    dest.append(converted)
            elif type(obj) == type({}):
                for key, value in dest.items():
                    converted = _convert_one(value, nextlevel)
                    dest[key] = converted
            else:  # struct or object
                dest["_type"] = type(obj)
                for propname in dir(obj):
                    _token = struct()
                    value = getattr(obj, propname, _token)
                    if value == _token:
                        continue  # Native methods are not inspectable. Ignore.
                    converted = _convert_one(value, nextlevel)
                    dest[propname] = converted
                if type(obj) == "Target":
                    if JavaInfo in obj:
                        dest["JavaInfo"] = _convert_one(obj[JavaInfo], nextlevel)
                    if CcInfo in obj:
                        dest["CcInfo"] = _convert_one(obj[CcInfo], nextlevel)

        queue = nextlevel
    return root

def _convert_one(val, nextlevel):
    nest = nextlevel != None
    if _is_sequence(val) and nest:
        out = []
        nextlevel.append((out, val))
        return out
    elif _is_atom(val) or not nest:
        return val
    elif type(val) == "File":
        return val.path
    elif type(val) == "Label":
        return str(val)
    else:  # by default try to convert object to dict
        out = {}
        nextlevel.append((out, val))
        return out

def _is_sequence(val):
    return _is_list(val) or _is_depset(val)

def _is_list(val):
    return type(val) == type([])

def _is_depset(val):
    return type(val) == type(depset())

def _is_atom(val):
    return (type(val) == type("") or
            type(val) == type(0) or
            type(val) == type(False) or
            type(val) == type(None))
