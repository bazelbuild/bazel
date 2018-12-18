// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.python;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.ensureDefaultIsPY2;

import com.google.common.base.Joiner;
import org.junit.Test;

/** Tests that are common to {@code py_binary} and {@code py_test}. */
public abstract class PyExecutableConfiguredTargetTestBase extends PyBaseConfiguredTargetTestBase {

  private final String ruleName;

  protected PyExecutableConfiguredTargetTestBase(String ruleName) {
    super(ruleName);
    this.ruleName = ruleName;
  }

  /**
   * Sets the configuration, then asserts that a configured target has the given Python version.
   *
   * <p>The configuration is given as a series of "--flag=value" strings.
   */
  protected void assertPythonVersionIs_UnderNewConfig(
      String targetName, PythonVersion version, String... flags) throws Exception {
    useConfiguration(flags);
    assertThat(getPythonVersion(getConfiguredTarget(targetName))).isEqualTo(version);
  }

  /**
   * Asserts that a configured target has the given Python version under multiple configurations.
   *
   * <p>The configurations are given as a series of arrays of "--flag=value" strings.
   *
   * <p>This destructively changes the current configuration.
   */
  protected void assertPythonVersionIs_UnderNewConfigs(
      String targetName, PythonVersion version, String[]... configs) throws Exception {
    for (String[] config : configs) {
      useConfiguration(config);
      assertWithMessage(String.format("Under config '%s'", Joiner.on(" ").join(config)))
          .that(getPythonVersion(getConfiguredTarget(targetName)))
          .isEqualTo(version);
    }
  }

  private String ruleDeclWithVersionAttr(String name, String version) {
    return ruleName + "(\n"
        + "    name = '" + name + "',\n"
        + "    srcs = ['" + name + ".py'],\n"
        + "    default_python_version = '" + version + "'\n"
        + ")\n";
  }

  @Test
  public void versionAttr_UnknownValue() throws Exception {
    checkError("pkg", "foo",
        // error:
        "invalid value in 'default_python_version' attribute: "
            + "has to be one of 'PY2' or 'PY3' instead of 'doesnotexist'",
        // build file:
        ruleDeclWithVersionAttr("foo", "doesnotexist"));
  }

  @Test
  public void versionAttr_BadValue() throws Exception {
    checkError("pkg", "foo",
        // error:
        "invalid value in 'default_python_version' attribute: "
            + "has to be one of 'PY2' or 'PY3' instead of 'PY2AND3'",
        // build file:
        ruleDeclWithVersionAttr("foo", "PY2AND3"));
  }

  @Test
  public void versionAttr_GoodValue() throws Exception {
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY2"));
    getConfiguredTarget("//pkg:foo");
    assertNoEvents();
  }

  @Test
  public void versionAttrWorksUnderOldAndNewSemantics_WhenNotDefaultValue() throws Exception {
    ensureDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY3"));

    assertPythonVersionIs_UnderNewConfigs(
        "//pkg:foo",
        PythonVersion.PY3,
        new String[] {"--experimental_better_python_version_mixing=false"},
        new String[] {"--experimental_better_python_version_mixing=true"});
  }

  @Test
  public void versionAttrWorksUnderOldAndNewSemantics_WhenSameAsDefaultValue() throws Exception {
    ensureDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY2"));

    assertPythonVersionIs_UnderNewConfigs(
        "//pkg:foo",
        PythonVersion.PY2,
        new String[] {"--experimental_better_python_version_mixing=false"},
        new String[] {"--experimental_better_python_version_mixing=true"});
  }

  @Test
  public void flagTakesPrecedenceUnderOldSemantics_NonDefaultValue() throws Exception {
    ensureDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY2"));
    assertPythonVersionIs_UnderNewConfig(
        "//pkg:foo",
        PythonVersion.PY3,
        "--experimental_better_python_version_mixing=false",
        "--force_python=PY3");
  }

  @Test
  public void flagTakesPrecedenceUnderOldSemantics_DefaultValue() throws Exception {
    ensureDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY3"));
    assertPythonVersionIs_UnderNewConfig(
        "//pkg:foo",
        PythonVersion.PY2,
        "--experimental_better_python_version_mixing=false",
        "--force_python=PY2");
  }

  @Test
  public void versionAttrTakesPrecedenceUnderNewSemantics_NonDefaultValue() throws Exception {
    ensureDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY3"));

    // Test against both flags.
    assertPythonVersionIs_UnderNewConfigs(
        "//pkg:foo",
        PythonVersion.PY3,
        new String[] {"--experimental_better_python_version_mixing=true", "--force_python=PY2"},
        new String[] {"--experimental_better_python_version_mixing=true", "--python_version=PY2"});
  }

  @Test
  public void versionAttrTakesPrecedenceUnderNewSemantics_DefaultValue() throws Exception {
    ensureDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithVersionAttr("foo", "PY2"));

    // Test against both flags.
    assertPythonVersionIs_UnderNewConfigs(
        "//pkg:foo",
        PythonVersion.PY2,
        new String[] {"--experimental_better_python_version_mixing=true", "--force_python=PY3"},
        new String[] {"--experimental_better_python_version_mixing=true", "--python_version=PY3"});
  }

  @Test
  public void canBuildWithDifferentVersionAttrs_UnderOldAndNewSemantics() throws Exception {
    scratch.file(
        "pkg/BUILD",
        ruleDeclWithVersionAttr("foo_v2", "PY2"),
        ruleDeclWithVersionAttr("foo_v3", "PY3"));

    assertPythonVersionIs_UnderNewConfigs(
        "//pkg:foo_v2",
        PythonVersion.PY2,
        new String[] {"--experimental_better_python_version_mixing=false"},
        new String[] {"--experimental_better_python_version_mixing=true"});
    assertPythonVersionIs_UnderNewConfigs(
        "//pkg:foo_v3",
        PythonVersion.PY3,
        new String[] {"--experimental_better_python_version_mixing=false"},
        new String[] {"--experimental_better_python_version_mixing=true"});
  }

  @Test
  public void canBuildWithDifferentVersionAttrs_UnderOldSemantics_FlagSetToDefault()
      throws Exception {
    ensureDefaultIsPY2();
    scratch.file(
        "pkg/BUILD",
        ruleDeclWithVersionAttr("foo_v2", "PY2"),
        ruleDeclWithVersionAttr("foo_v3", "PY3"));

    assertPythonVersionIs_UnderNewConfig("//pkg:foo_v2", PythonVersion.PY2, "--force_python=PY2");
    assertPythonVersionIs_UnderNewConfig("//pkg:foo_v3", PythonVersion.PY2, "--force_python=PY2");
  }

  @Test
  public void canBuildWithDifferentVersionAttrs_UnderOldSemantics_FlagSetToNonDefault()
      throws Exception {
    ensureDefaultIsPY2();
    scratch.file(
        "pkg/BUILD",
        ruleDeclWithVersionAttr("foo_v2", "PY2"),
        ruleDeclWithVersionAttr("foo_v3", "PY3"));

    assertPythonVersionIs_UnderNewConfig("//pkg:foo_v2", PythonVersion.PY3, "--force_python=PY3");
    assertPythonVersionIs_UnderNewConfig("//pkg:foo_v3", PythonVersion.PY3, "--force_python=PY3");
  }

  @Test
  public void srcsVersionClashesWithVersionAttr() throws Exception {
    checkError("pkg", "foo",
        // error:
        "'//pkg:foo' can only be used with Python 2",
        // build file:
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        "    srcs_version = 'PY2ONLY',",
        "    default_python_version = 'PY3')");
  }

  @Test
  public void srcsVersionClashesWithVersionAttr_Implicitly() throws Exception {
    ensureDefaultIsPY2(); // When changed to PY3, flip srcs_version below to be PY2ONLY.
    // Fails because default_python_version is PY2 by default, so the config is set to PY2
    // regardless of srcs_version.
    checkError("pkg", "foo",
        // error:
        "'//pkg:foo' can only be used with Python 3",
        // build file:
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        "    srcs_version = 'PY3ONLY')");
  }
}
