# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Definition of incoming apple crosstool transition."""

transition = _builtins.toplevel.transition

def _cpu_string(platform_type, settings):
    apple_split_cpu = settings["//command_line_option:apple_split_cpu"]
    if apple_split_cpu:
        if platform_type == "macos":
            return "darwin_{}".format(apple_split_cpu)
        return "{}_{}".format(platform_type, apple_split_cpu)
    return settings["//command_line_option:cpu"]

def _output_dictionary(settings, cpu, platform_type, platforms):
    return {
        "//command_line_option:apple configuration distinguisher": "applebin_" + platform_type,
        "//command_line_option:compiler": None,
        "//command_line_option:cpu": cpu,
        "//command_line_option:crosstool_top": (
            settings["//command_line_option:apple_crosstool_top"]
        ),
        "//command_line_option:platforms": platforms,
        "//command_line_option:fission": [],
        "//command_line_option:grte_top": settings["//command_line_option:apple_grte_top"],
    }

def _apple_crosstool_transition_impl(settings, attr):
    platform_type = str(settings["//command_line_option:apple_platform_type"])
    cpu = _cpu_string(platform_type, settings)
    if settings["//command_line_option:incompatible_enable_apple_toolchain_resolution"]:
        platforms = (
            settings["//command_line_option:apple_platforms"] or
            settings["//command_line_option:platforms"]
        )
        return _output_dictionary(settings, cpu, platform_type, platforms)
    crosstools_are_equal = (
        settings["//command_line_option:crosstool_top"] ==
        settings["//command_line_option:apple_crosstool_top"]
    )
    if cpu == settings["//command_line_option:cpu"] and crosstools_are_equal:
        # No changes necessary.
        return {}

    # Ensure platforms aren't set so that platform mapping can take place.
    return _output_dictionary(settings, cpu, platform_type, [])

_apple_rule_base_transition_inputs = [
    "//command_line_option:apple configuration distinguisher",
    "//command_line_option:apple_platform_type",
    "//command_line_option:apple_platforms",
    "//command_line_option:apple_crosstool_top",
    "//command_line_option:crosstool_top",
    "//command_line_option:apple_split_cpu",
    "//command_line_option:apple_grte_top",
    "//command_line_option:cpu",
    "//command_line_option:ios_multi_cpus",
    "//command_line_option:macos_cpus",
    "//command_line_option:tvos_cpus",
    "//command_line_option:visionos_cpus",
    "//command_line_option:watchos_cpus",
    "//command_line_option:catalyst_cpus",
    "//command_line_option:platforms",
    "//command_line_option:fission",
    "//command_line_option:grte_top",
    "//command_line_option:incompatible_enable_apple_toolchain_resolution",
]
_apple_rule_base_transition_outputs = [
    "//command_line_option:apple configuration distinguisher",
    "//command_line_option:compiler",
    "//command_line_option:cpu",
    "//command_line_option:crosstool_top",
    "//command_line_option:platforms",
    "//command_line_option:fission",
    "//command_line_option:grte_top",
]

apple_crosstool_transition = transition(
    implementation = _apple_crosstool_transition_impl,
    inputs = _apple_rule_base_transition_inputs,
    outputs = _apple_rule_base_transition_outputs,
)
