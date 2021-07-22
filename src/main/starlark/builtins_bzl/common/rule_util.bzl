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

def create_rule(impl, attrs = {}, deps = [], fragments = [], **kwargs):
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
        **kwargs: extra args to be passed for rule creation

    Returns:
        The composed rule
    """
    merged_attrs = dict()
    fragments = list(fragments)
    for dep in deps:
        merged_attrs.update(dep.attrs)
        fragments.extend(dep.fragments)
    merged_attrs.update(attrs)

    return rule(
        implementation = impl,
        attrs = merged_attrs,
        fragments = fragments,
        **kwargs
    )

def create_dep(call, attrs = {}, fragments = []):
    """Combines a dependency's executable function, attributes, and fragments.

    Args:
        call: the executable function
        attrs: dict of required rule attrs
        fragments: list of required configuration fragments

    Returns:
        The struct
    """
    return _create_dep(call, attrs, fragments)

def _create_dep(call, attrs = {}, fragments = []):
    return struct(
        call = call,
        attrs = attrs,
        fragments = fragments,
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
    for dep in deps:
        merged_attrs.update(dep.attrs)
        merged_frags.extend(dep.fragments)

    return _create_dep(
        call = merge_func,
        attrs = merged_attrs,
        fragments = merged_frags,
    )
