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

from absl.testing import absltest
from src.test.py.bazel import test_base


class OptionsTest(test_base.TestBase):

  def testCanOverrideStarlarkFlagInBazelrcConfigStanza(self):
    self.ScratchFile("MODULE.bazel")
    self.ScratchFile(
        "bazelrc",
        [
            "build:red --//f:color=red",
        ],
    )
    self.ScratchFile(
        "f/BUILD.bazel",
        [
            'load(":f.bzl", "color", "r")',
            "color(",
            '    name = "color",',
            '    build_setting_default = "white",',
            ")",
            'r(name = "r")',
        ],
    )
    self.ScratchFile(
        "f/f.bzl",
        [
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
        ],
    )

    _, _, stderr = self.RunBazel([
        "--bazelrc=bazelrc",
        "build",
        "--nobuild",
        "//f:r",
        "--config=red",
        "--//f:color=green",
    ])
    self.assertTrue(
        any("/f/f.bzl:9:10: green" in line for line in stderr),
        "\n".join(stderr),
    )

    _, _, stderr = self.RunBazel([
        "--bazelrc=bazelrc",
        "build",
        "--nobuild",
        "//f:r",
        "--//f:color=green",
        "--config=red",
    ])
    self.assertTrue(
        any("/f/f.bzl:9:10: red" in line for line in stderr),
        "\n".join(stderr),
    )

  def testCommonPseudoCommand(self):
    self.ScratchFile("MODULE.bazel")
    self.ScratchFile(
        ".bazelrc",
        [
            "common --copt=-Dfoo",
            "common --copt -Dbar",
            "common:my-config --copt=-Dbaz",
            "common:my-config --copt -Dquz",
        ],
    )
    self.ScratchFile(
        "pkg/BUILD.bazel",
        [
            "cc_binary(name='main',srcs=['main.cc'])",
        ],
    )
    self.ScratchFile(
        "pkg/main.cc",
        [
            "#include <stdio.h>",
            "int main() {",
            "#ifdef foo",
            '  printf("foo\\n");',
            "#endif",
            "#ifdef bar",
            '  printf("bar\\n");',
            "#endif",
            "#ifdef baz",
            '  printf("baz\\n");',
            "#endif",
            "#ifdef quz",
            '  printf("quz\\n");',
            "#endif",
            "  return 0;",
            "}",
        ],
    )

    # Check that run honors the common flags.
    _, stdout, stderr = self.RunBazel([
        "run",
        "--announce_rc",
        "//pkg:main",
    ])
    self.assertEqual(
        ["foo", "bar"],
        stdout,
    )
    self.assertNotRegex(
        "\n".join(stderr),
        "Ignored as unsupported",
    )

    _, stdout, stderr = self.RunBazel([
        "run",
        "--announce_rc",
        "--config=my-config",
        "//pkg:main",
    ])
    self.assertEqual(
        ["foo", "bar", "baz", "quz"],
        stdout,
    )
    self.assertNotRegex(
        "\n".join(stderr),
        "Ignored as unsupported",
    )

    # Check that query ignores the unsupported common flags.
    _, stdout, stderr = self.RunBazel([
        "query",
        "--announce_rc",
        "//pkg:main",
    ])
    self.assertRegex(
        "\n".join(stderr),
        "Ignored as unsupported by 'query': --copt=-Dfoo --copt -Dbar",
    )
    self.assertNotRegex(
        "\n".join(stderr),
        "Ignored as unsupported by 'query': --copt=-Dbaz --copt -Dquz",
    )

    _, stdout, stderr = self.RunBazel([
        "query",
        "--announce_rc",
        "--config=my-config",
        "//pkg:main",
    ])
    self.assertRegex(
        "\n".join(stderr),
        "Ignored as unsupported by 'query': --copt=-Dfoo --copt -Dbar",
    )
    self.assertRegex(
        "\n".join(stderr),
        "Ignored as unsupported by 'query': --copt=-Dbaz --copt -Dquz",
    )

  def testCommonPseudoCommand_singleLineParsesUnambiguously(self):
    self.ScratchFile("MODULE.bazel")
    self.ScratchFile(
        ".bazelrc",
        [
            # First and third option are ignored by build, but valid options for
            # cquery. The first one expects no value, the third one does.
            "common --implicit_deps --copt=-Dfoo --output files --copt=-Dbar",
        ],
    )
    self.ScratchFile(
        "pkg/BUILD.bazel",
        [
            "cc_binary(name='main',srcs=['main.cc'])",
        ],
    )
    self.ScratchFile(
        "pkg/main.cc",
        [
            "#include <stdio.h>",
            "int main() {",
            "#ifdef foo",
            '  printf("foo\\n");',
            "#endif",
            "#ifdef bar",
            '  printf("bar\\n");',
            "#endif",
            "  return 0;",
            "}",
        ],
    )

    # Check that run honors the common flags.
    _, stdout, _ = self.RunBazel([
        "run",
        "//pkg:main",
    ])
    self.assertEqual(
        ["foo", "bar"],
        stdout,
    )

  def testCommonPseudoCommand_unsupportedOptionValue(self):
    self.ScratchFile("MODULE.bazel")
    self.ScratchFile(
        ".bazelrc",
        [
            "common --output=starlark",
        ],
    )
    self.ScratchFile(
        "pkg/BUILD.bazel",
        [
            "cc_binary(name='main',srcs=['main.cc'])",
        ],
    )

    # Check that cquery honors the common flag.
    _, stdout, _ = self.RunBazel([
        "cquery",
        "--starlark:expr=target.label.name",
        "//pkg:main",
    ])
    self.assertEqual(
        ["main"],
        stdout,
    )

    # Check that query fails as it supports the --output flag, but not its
    # value.
    exit_code, stdout, stderr = self.RunBazel(
        [
            "query",
            "//pkg:main",
        ],
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 2, stderr)
    self.assertTrue(
        any(
            "ERROR: Invalid output format 'starlark'." in line
            for line in stderr
        ),
        stderr,
    )

  def testCommonPseudoCommand_allowResidueFalseCommandIgnoresStarlarkOptions(
      self,
  ):
    self.ScratchFile("MODULE.bazel")
    self.ScratchFile(
        ".bazelrc",
        [
            "common --@foo//bar:flag",
        ],
    )

    # Check that version doesn't fail.
    self.RunBazel(["version"])

  def testConfigExpansion_failsOnUnsupportedFlag(self):
    self.ScratchFile("MODULE.bazel")
    self.ScratchFile("BUILD.bazel")
    self.ScratchFile(
        ".bazelrc",
        [
            "build:abc --copt",
            "build:abc -DFOO",
            "common:abc --noverbose_test_summary",
            "common:abc --verbose_test_summary",
            "build:abc --copt",
            "build:abc -DFOO",
            "build:abc --verbose_test_summary",
            "common:def --config=abc",
        ],
    )

    exit_code, _, stderr = self.RunBazel(
        ["build", "--announce_rc", "--config=def", "//..."], allow_failure=True
    )
    self.AssertExitCode(exit_code, 2, stderr)
    self.assertIn(
        "ERROR: --verbose_test_summary :: Unrecognized option:"
        " --verbose_test_summary",
        stderr,
    )


if __name__ == "__main__":
  absltest.main()
