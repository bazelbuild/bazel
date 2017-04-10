# Copyright 2017 The Bazel Authors. All rights reserved.
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
"""Useful functions and rules for defining toolchains."""

_default_toolchain_rule_attrs = {
    "exec_compatible_with": attr.label_list(
        providers = [platform_common.ConstraintValueInfo]),
    "target_compatible_with": attr.label_list(
        providers = [platform_common.ConstraintValueInfo]),
}

def default_toolchain_rule_impl(ctx, override_attrs = {}):
  """A default implementation for toolchain_rule.
  This implementation creates a toolchain provider and adds all extra
  attributes.

  Args:
    ctx: The rule context.
    override_attrs: Any data in this dict will override the corresponding
      attribute from the context. toolchain_rule implementations can use this
      to customize the values set in the provider.

  Returns:
    The created toolchain provider.
  """
  toolchain_data = {}

  # Collect the extra_args from ctx.attrs.
  attr_names = ctx.attr._extra_attr_names
  for name in attr_names:
    if name in override_attrs:
      attr = override_attrs[name]
    else:
      attr = getattr(ctx.attr, name)
    toolchain_data[name] = attr

  toolchain = platform_common.toolchain(
      exec_compatible_with = ctx.attr.exec_compatible_with,
      target_compatible_with = ctx.attr.target_compatible_with,
      **toolchain_data)

  return [toolchain]

def toolchain_rule(implementation = default_toolchain_rule_impl, fragments = [], extra_attrs = {}):
  return rule(
      implementation = implementation,
      attrs = _default_toolchain_rule_attrs + extra_attrs + {
          # default_toolchain_rule_impl needs this to know what attributes are extra args.
          "_extra_attr_names": attr.string_list(default = extra_attrs.keys()),
      },
      fragments = fragments,
  )
