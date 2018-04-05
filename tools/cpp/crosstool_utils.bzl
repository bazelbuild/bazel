# pylint: disable=g-bad-file-header
# Copyright 2018 The Bazel Authors. All rights reserved.
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
"""Utility functions for writing crosstool files in Skylark"""

# All possible C++ compile actions
COMPILE_ACTIONS = [
    "c-compile",
    "c++-compile",
    "c++-header-parsing",
    "c++-header-preprocessing",
    "c++-module-compile",
    "c++-module-codegen",
    "assemble",
    "preprocess-assemble",
    "clif-match",
    "linkstamp-compile",
    "cc-flags-make-variable",
]

# All possible C++ link actions
LINK_ACTIONS = [
    "c++-link-dynamic-library",
    "c++-link-nodeps-dynamic-library",
    "c++-link-executable",
]

# All possible C++ archive actions
ARCHIVE_ACTIONS = [
    "c++-link-static-library",
]

# All remaining actions used by C++ rules that are configured in the CROSSTOOL
OTHER_ACTIONS = [
    "strip"
]


def action_config(action_name, tool_path):
  """Emit action_config message.

  Examples:
    action_config("c-compile", "/usr/bin/gcc") ->
      action_config {
        config_name: 'c-compile'
        action_name: 'c-compile'
        tool {
          tool_path: '/usr/bin/gcc'
        }
      }

  Args:
    action_name: name of the action
    tool_path: absolute or CROSSTOOL-relative path to the tool

  Returns:
    a string to be placed into the CROSSTOOL
  """
  if action_name == None or action_name == "":
    fail("action_name must be present")
  if tool_path == None or tool_path == "":
    fail("tool_path must be present")
  return """
  action_config {{
    config_name: '{action_name}'
    action_name: '{action_name}'
    tool {{
      tool_path: '{tool_path}'
    }}
  }}""".format(action_name=action_name, tool_path=tool_path)


def feature(name, flag_sets, enabled = True, provides = None):
  """Emit feature message.

  Examples:
    feature("fully_static_link", flag_sets, enabled = False) ->
      feature {
        name: 'fully_static_link'
        enabled = false
        <flags_sets>
      }

  Args:
    name: name of the feature
    flag_sets: a collection of flag_set messages
    enabled: whether this feature is turned on by default
    provides: a symbol this feature provides, used to implement mutually incompatible features

  Returns:
    a string to be placed into the CROSSTOOL
  """
  if name == None or name == "":
    fail("feature name must be present")
  return """
  feature {{
    name: '{name}'
    enabled: {enabled}{provides}{flag_sets}
  }}""".format(
       provides=("\n      provides: '%s'" % provides if provides != None else ""),
       name=name,
       enabled=_to_proto_value(enabled),
       flag_sets="".join(flag_sets))


def simple_feature(name, actions, flags, enabled = True, provides = None,
                   expand_if_all_available = [], iterate_over = None):
  """Sugar for emitting simple feature message.

  Examples:
    simple_feature("foo", ['c-compile'], flags("-foo")) ->
      feature {
        name: 'foo'
        flag_set {
          action: 'c-compile'
          flag_group {
            flag: '-foo'
          }
        }
      }

  Args:
    name: name of the feature
    actions: for which actions should flags be emitted
    flags: a collection of flag messages
    enabled: whether this feature is turned on by default
    provides: a symbol this feature provides, used to implement mutually incompatible features
    expand_if_all_available: specify which build variables need to be present
      for this group to be expanded
    iterate_over: expand this flag_group for every item in the build variable

  Returns:
    a string to be placed into the CROSSTOOL
  """
  if len(flags) == 0:
    return feature(name, [])
  else:
    return feature(
        name,
        [flag_set(
            actions,
            [flag_group(
                [flag(f) for f in flags],
                iterate_over=iterate_over,
                expand_if_all_available=expand_if_all_available)])],
        enabled = enabled,
        provides = provides)


def flag_set(actions, flag_groups):
  """Emit flag_set message.

  Examples:
    flag_set(['c-compile'], flag_groups) ->
      flag_set {
        action: 'c-compile'
        <flag_groups>
      }

  Args:
    actions: for which actions should flags be emitted
    flag_groups: a collection of flag_group messages

  Returns:
    a string to be placed into the CROSSTOOL
  """
  if actions == None or len(actions) == 0:
    fail("empty actions list is not allowed for flag_set")
  if flag_groups == None or len(flag_groups) == 0:
    fail("empty flag_groups list is not allowed for flag_set")
  actions_string = ""
  for action in actions: actions_string += "\n      action: '%s'" % action

  return """
    flag_set {{{actions}{flag_groups}
    }}""".format(actions=actions_string, flag_groups="".join(flag_groups))


def flag_group(
    content, expand_if_all_available = [], expand_if_none_available = [], expand_if_true = [],
    expand_if_false = [], expand_if_equal = [], iterate_over = None):
  """Emit flag_group message.

  Examples:
    flag_group(flags("-foo %{output_file}"), expand_if_all_available="output_file") ->
      flag_group { expand_if_all_available: "output_file"
        flag: "-foo %{output_file}"
      }

  Args:
    content: a collection of flag messages or a collection of flag_group messages
    expand_if_all_available: specify which build variables need to be present
      for this group to be expanded
    expand_if_none_available: specify which build variables need to be missing
      for this group to be expanded
    expand_if_true: specify which build variables need to be truthy for this group
      to be expanded
    expand_if_false: specify which build variables need to be falsey for this group
      to be expanded
    expand_if_equal: [[var1, value1], [var2, value2]...] specify what values
      should specific build variables have for this group to be expanded
    iterate_over: expand this flag_group for every item in the build variable

  Returns:
    a string to be placed into the CROSSTOOL
  """
  if content == None or len(content)== 0:
    fail("flag_group without flags is not allowed")
  conditions = ""
  for var in expand_if_all_available:
    conditions += "\n        expand_if_all_available: '%s'" % var
  for var in expand_if_none_available:
    conditions += "\n        expand_if_none_available: '%s'" % var
  for var in expand_if_true:
    conditions += "\n        expand_if_true: '%s'" % var
  for var in expand_if_false:
    conditions += "\n        expand_if_false: '%s'" % var
  for var in expand_if_equal:
    conditions += "\n        expand_if_equal { variable: '%s' value: '%s' }" % (var[0], var[1])
  return """
      flag_group {{{conditions}{iterate_over}{content}
      }}""".format(
      content="".join(content),
      iterate_over=("\n        iterate_over: '%s'" % iterate_over if iterate_over != None else ""),
      conditions=conditions)


def flag(flag):
  """Emit flag field.

  Examples:
    flag("-foo") -> flag: '-foo'

  Args:
    flag: value to be emitted to the command line

  Returns:
    a string to be placed into the CROSSTOOL
  """
  return "\n        flag: '%s'" % flag


def flags(*flags):
  """Sugar for emitting sequence of flag fields.

  Examples:
    flags("-foo", "-bar") ->
      flag: '-foo'
      flag: '-bar'

  Args:
    *flags: values to be emitted to the command line

  Returns:
    a string to be placed into the CROSSTOOL
  """
  return [flag(f) for f in flags]


def _to_proto_value(boolean):
  if boolean:
    return "true"
  else:
    return "false"
