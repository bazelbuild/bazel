# Copyright 2016 The Bazel Authors. All rights reserved.
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

"""Common definitions for Apple rules."""

APPLE_SIMULATOR_ARCHITECTURES = ["i386", "x86_64"]
"""Architectures that are used by the simulator (iOS, tvOS and watchOS)."""

IOS_DEVICE_ARCHITECTURES = ["armv6", "armv7", "arm64"]
"""Architectures that are used by iOS devices."""

TVOS_DEVICE_ARCHITECTURES  = ["arm64"]
"""Architectures that are used by tvOS devices."""

WATCHOS_DEVICE_ARCHITECTURES = ["armv7k"]
"""Architectures that are used by watchOS devices."""

APPLE_DEFAULT_ARCHITECTURES = ["x86_64", "arm64", "armv7k"]
"""Architectures commonly used for building/testing on simulators/devices."""

APPLE_FRAGMENTS = ["apple"]
"""Configuration fragments containing Apple specific information."""

DARWIN_EXECUTION_REQUIREMENTS = {"requires-darwin": ""}
"""Standard execution requirements to force building on Mac.

See :func:`apple_action`."""

XCRUNWRAPPER_LABEL = "//external:xcrunwrapper"
"""The label for xcrunwrapper tool."""

def module_cache_path(ctx):
  """Returns the Clang module cache path to use for this rule."""
  return ctx.configuration.genfiles_dir.path + "/_objc_module_cache"

def apple_action(ctx, **kw):
  """Creates an action that only runs on MacOS/Darwin.

  Call it similar to how you would call ctx.action:
    apple_action(ctx, outputs=[...], inputs=[...],...)
  """
  execution_requirements = kw.get('execution_requirements', {})
  execution_requirements += DARWIN_EXECUTION_REQUIREMENTS
  kw['execution_requirements'] = execution_requirements

  ctx.action(**kw)

def xcrun_action(ctx, **kw):
  """Creates an apple action that executes xcrunwrapper.

  args:
    ctx: The context of the rule that owns this action.

  This method takes the same keyword arguments as ctx.action, however you don't
  need to specify the executable.
  """
  platform = ctx.fragments.apple.ios_cpu_platform()
  action_env = ctx.fragments.apple.target_apple_env(platform) \
      + ctx.fragments.apple.apple_host_system_env()
  env = kw.get('env', {})
  kw['env'] = env + action_env

  apple_action(ctx, executable=ctx.executable._xcrunwrapper, **kw)
