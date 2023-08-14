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

"""Defines rule utilities"""

def merge_attrs(*attribute_dicts, override_attrs = {}, remove_attrs = []):
    """Merges attributes together.

    Attributes are first merged, then overridden and removed.

    If there are duplicate definitions of an attribute, the last one is used.
    (Current API doesn't let us compare)

    Overridden and removed attributes need to be present.

    Args:
      *attribute_dicts: (*dict[str,Attribute]) A list of attribute dictionaries
        to merge together.
      override_attrs: (dict[str,Attribute]) A dictionary of attributes to override
      remove_attrs: (list[str]) A list of attributes to remove.
    Returns:
      (dict[str,Attribute]) The merged attributes dictionary.
    """
    all_attributes = {}
    for attribute_dict in attribute_dicts:
        for key, attr in attribute_dict.items():
            all_attributes.setdefault(key, attr)
    for key, attr in override_attrs.items():
        if all_attributes.get(key) == None:
            fail("Trying to override attribute %s where there is none." % key)
        all_attributes[key] = attr
    for key in remove_attrs:
        if key in override_attrs:
            fail("Trying to remove overridden attribute %s." % key)
        if key not in all_attributes:
            fail("Trying to remove non-existent attribute %s." % key)
        all_attributes.pop(key)
    return all_attributes
