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

"""Defines create_rule and create_dep macros"""

def create_rule(impl, attrs = {}, deps = [], fragments = [], remove_attrs = [], **kwargs):
    """Creates a rule composed from dependencies.

    Args:
        impl: The implementation function of the rule, taking as parameters the
            rule ctx followed by the executable function of each dependency
        attrs: Dict of attributes required by the rule. These will override any
            conflicting attributes specified by dependencies
        deps: Dict of name->dependency mappings, with each dependency struct
            created using 'create_dep'. The keys of this dict are the parameter
            names received by 'impl'
        fragments: List of configuration fragments required by the rule
        remove_attrs: List of attributes to remove from the implementation.
        **kwargs: extra args to be passed for rule creation

    Returns:
        The composed rule
    """
    merged_attrs = dict()
    fragments = list(fragments)
    merged_mandatory_attrs = []

    for dep in deps:
        merged_attrs.update(dep.attrs)
        fragments.extend(dep.fragments)
        merged_mandatory_attrs.extend(dep.mandatory_attrs)
    merged_attrs.update(attrs)

    for attr in remove_attrs:
        if attr in merged_mandatory_attrs:
            fail("Cannot remove mandatory attribute %s" % attr)
        merged_attrs.pop(attr)

    return rule(
        implementation = impl,
        attrs = merged_attrs,
        fragments = fragments,
        **kwargs
    )

def create_dep(call, attrs = {}, fragments = [], mandatory_attrs = None):
    """Combines a dependency's executable function, attributes, and fragments.

    Args:
        call: the executable function
        attrs: dict of required rule attrs
        fragments: list of required configuration fragments
        mandatory_attrs: list of attributes that can't be removed later
          (when not set, all attributes are mandatory)
    Returns:
        The struct
    """
    return _create_dep(call, attrs, fragments, mandatory_attrs if mandatory_attrs else attrs.keys())

def _create_dep(call, attrs = {}, fragments = [], mandatory_attrs = []):
    return struct(
        call = call,
        attrs = attrs,
        fragments = fragments,
        mandatory_attrs = mandatory_attrs,
    )

def create_composite_dep(merge_func, *deps):
    """Creates a dependency struct from multiple dependencies

    Args:
        merge_func: The executable function to evaluate the dependencies.
        *deps: The dependencies to compose provided as keyword args

    Returns:
        A dependency struct
    """
    merged_attrs = dict()
    merged_frags = []
    merged_mandatory_attrs = []
    for dep in deps:
        merged_attrs.update(dep.attrs)
        merged_frags.extend(dep.fragments)
        merged_mandatory_attrs.extend(dep.mandatory_attrs)

    return _create_dep(
        call = merge_func,
        attrs = merged_attrs,
        fragments = merged_frags,
        mandatory_attrs = merged_mandatory_attrs,
    )

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
