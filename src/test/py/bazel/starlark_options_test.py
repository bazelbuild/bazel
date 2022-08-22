# Copyright 2022 The Bazel Authors. All rights reserved.
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

import unittest

from src.test.py.bazel import test_base


class StarlarkOptionsTest(test_base.TestBase):

  def testCanOverrideStarlarkFlagInBazelrcConfigStanza(self):
    self.ScratchFile("WORKSPACE.bazel")
    self.ScratchFile("bazelrc", [
        "build:red --//f:color=red",
    ])
    self.ScratchFile("f/BUILD.bazel", [
        'load(":f.bzl", "color", "r")',
        "color(",
        '    name = "color",',
        '    build_setting_default = "white",',
        ")",
        'r(name = "r")',
    ])
    self.ScratchFile("f/f.bzl", [
        'ColorValue = provider("color")',
        "def _color_impl(ctx):",
        "    return [ColorValue(color = ctx.build_setting_value)]",
        "color = rule(",
        "    implementation = _color_impl,",
        "build_setting = config.string(flag = True),",
        ")",
        "def _r_impl(ctx):",
        "    print(ctx.attr._color[ColorValue].color)",
        "    return [DefaultInfo()]",
        "r = rule(",
        "    implementation = _r_impl,",
        '    attrs = {"_color": attr.label(default = "//f:color")},',
        ")",
    ])

    exit_code, _, stderr = self.RunBazel([
        "--bazelrc=bazelrc",
        "build",
        "--nobuild",
        "//f:r",
        "--config=red",
        "--//f:color=green",
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue(
        any("/f/f.bzl:9:10: green" in line for line in stderr),
        "\n".join(stderr),
    )

    exit_code, _, stderr = self.RunBazel([
        "--bazelrc=bazelrc",
        "build",
        "--nobuild",
        "//f:r",
        "--//f:color=green",
        "--config=red",
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    self.assertTrue(
        any("/f/f.bzl:9:10: red" in line for line in stderr),
        "\n".join(stderr),
    )


if __name__ == "__main__":
  unittest.main()
