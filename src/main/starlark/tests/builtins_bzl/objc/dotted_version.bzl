# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Represents Xcode versions and allows parsing them.

<p>Xcode versions are formed of multiple components, separated by periods, for example
<code>4.5.6</code> or <code>5.0.1beta2</code>. Components must start with a non-negative integer
and at least one component must be present.

<p>Specifically, the format of a component is <code>\\d+([a-z0-9]*?)?(\\d+)?</code>.

<p>Dotted versions are ordered using natural integer sorting on components in order from first to
last where any missing element is considered to have the value 0 if they don't contain any
non-numeric characters. For example:

<pre>
  3.1.25 > 3.1.1
  3.1.20 > 3.1.2
  3.1.1 > 3.1
  3.1 == 3.1.0.0
  3.2 > 3.1.8
</pre>

<p>If the component contains any alphabetic characters after the leading integer, it is
considered <strong>smaller</strong> than any components with the same integer but larger than any
component with a smaller integer. If the integers are the same, the alphabetic sequences are
compared lexicographically, and if <i>they</i> turn out to be the same, the final (optional)
integer is compared. As with the leading integer, this final integer is considered to be 0 if not
present. For example:

<pre>
  3.1.1 > 3.1.1beta3
  3.1.1beta1 > 3.1.0
  3.1 > 3.1.0alpha1

  3.1.0beta0 > 3.1.0alpha5.6
  3.4.2alpha2 > 3.4.2alpha1
  3.4.2alpha2 > 3.4.2alpha1.5
  3.1alpha1 > 3.1alpha
</pre>

Xcode version stringss are parsed by creating DottedVersion objects from them, via the
dotted_version() function.
<p>
DottedVersion objects have one method: compare_to(other) which takes another DottedVersion
object as parameter and returns -1, 0 or 1 depending on whether the "self" object is smaller
than, equal to or greater than the "other" object.
<p>
The object is represented by a struct with these fields:
* member_vars: a struct with the object's member variables, as returned by
    _member_vars_from_string()
* compare_to: compare method with parameter other, containing member_vars as closure.
    self.compare_to(other) returns 0 on equality, -1 if self is smaller than other, 1 otherwise.

"""
# TODO: b/331163027 - Move this file to third_party/bazel/src/main/starlark/builtins_bzl/common/objc
# once it is ready to be referenced in apple_common.

def _component_from_string(component_string):
    """Constructs a component struct from a component string.

    This is where the heavy lifting of parsing version strings happens.

    Args:
      component_string: the component string

    Returns:
      a struct with these fields:
        * first_number: int, the number with which the component string starts
        * alpha_sequence: string, the optional alphanum sequence following the first number,
            parsed non-greedily
        * second_number: int, the optional number after the alphanum sequence
        * string_representation: the original component_string
    """
    if not component_string:
        fail("Component must not be empty")
    length = len(component_string)
    first_number_end = 0
    second_number_start = length
    for i in range(length):
        if component_string[i].isdigit():
            first_number_end = i + 1
        else:
            break
    if first_number_end == 0:
        fail("error in dotted_version.bzl: component " + component_string +
             " wasn't identified as descriptive component.")
    first_number = component_string[:first_number_end]
    if not first_number.isdigit():
        fail("error in dotted_version.bzl: in component " + component_string +
             " first_number " + first_number + " was identified which isn't numeric.")
    first_number = int(first_number)
    alpha_sequence = ""
    second_number = 0
    if first_number_end < length:
        for i in range(length):
            if component_string[length - i - 1].isdigit():
                second_number_start = length - i - 1
            else:
                break
        if second_number_start <= first_number_end:
            fail("error in dotted_version.bzl: in " + component_string + ", second_number_start = " +
                 str(second_number_start) + " <= first_number_end = " + str(first_number_end))
        alpha_sequence = component_string[first_number_end:second_number_start]
        if not alpha_sequence.isalnum():
            fail("alpha_sequence " + alpha_sequence + " in component " + component_string +
                 "must be alphanumeric.")
        if second_number_start < length:
            second_number = component_string[second_number_start:]
            if not second_number.isdigit():
                fail("error in dotted_version.bzl: in component " + component_string +
                     " second_number " + second_number + " was identified which isn't numeric.")
            second_number = int(second_number)
    return struct(
        first_number = first_number,
        alpha_sequence = alpha_sequence,
        second_number = second_number,
        string_representation = component_string,
    )

def _is_descriptive_component(component_string):
    """A component is considered descriptive if it starts with a letter."""
    return component_string.elems()[0].isalpha()

def _member_vars_from_string(version_string):
    """Constructs the object's data struct from the version string.

    The data struct contains these fields:
    * components: the version's components, from the version string split by periods, as a list of
        structs as returned by _component_from_string()
    * string_representation: the version string

    Ignoring descriptive components and everything after them happens at this stage.

    Args:
      version_string: the Xcode version as string

    Returns:
      the object's data struct
    """
    if not version_string:
        fail("Dotted version must not be empty")
    component_strings = version_string.split(".")
    components = []
    for s in component_strings:
        if _is_descriptive_component(s):
            break
        components.append(_component_from_string(s))
    return struct(
        components = components,
        string_representation = version_string,
    )

def _cmp(val1, val2):
    """Generic comparison function.

    Args:
      val1: int or string
      val2: int or string, must be the same type as val1

    Returns:
        0 if val1 == val2, -1 if val1 < val2, 1 if val1 > val2.
    """
    if val1 == val2:
        return 0
    if val1 < val2:
        return -1
    return 1

def _compare_components(component1, component2):
    """Component comparison function.

    Compares by first_number, on equality then by alpha_sequence, then by second_number.

    Args:
      component1: a struct as returned by _component_from_string()
      component2: a struct as returned by _component_from_string()

    Returns:
        0 on equality, -1 if component1 < component2, 1 if component1 > component2.
    """
    cmp_result = _cmp(component1.first_number, component2.first_number)
    if cmp_result:
        return cmp_result
    cmp_result = _cmp(component1.alpha_sequence, component2.alpha_sequence)
    if cmp_result:
        return cmp_result
    cmp_result = _cmp(component1.second_number, component2.second_number)
    if cmp_result:
        return cmp_result
    return 0

_NULL_COMPONENT = _component_from_string("0")

def _get_component(components, i):
    """Returns the i-th component from a list, or _NULL_COMPONENT if i is out of range."""
    if i >= len(components):
        return _NULL_COMPONENT
    return components[i]

def _compare_member_vars(member_vars1, member_vars2):
    """Compares two member_vars structs by their components.

    The shorter compontents list is on the fly filled up with _NULL_COMPONENT elements.

    Args:
      member_vars1: a member_vars struct
      member_vars2: a member_vars struct
    Returns:
        0 on equality, -1 if member_vars1 < member_vars2, 1 if member_vars1 > member_vars2.
    """
    max_length = max(len(member_vars1.components), len(member_vars2.components))
    for i in range(max_length):
        cmp_result = _compare_components(
            _get_component(member_vars1.components, i),
            _get_component(member_vars2.components, i),
        )
        if cmp_result:
            return cmp_result
    return 0

def dotted_version(version_string):
    """The constructor for the DottedVersion object.

    Args:
      version_string: the Xcode version as string from which to construct the DottedVersion object

    Returns:
      the DottedVersion object
    """
    member_vars = _member_vars_from_string(version_string)
    version_object = struct(
        member_vars = member_vars,
        compare_to = lambda other: _compare_member_vars(member_vars, other.member_vars),
    )
    return version_object
