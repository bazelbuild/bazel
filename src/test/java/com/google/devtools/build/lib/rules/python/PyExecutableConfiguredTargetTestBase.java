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
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.assumesDefaultIsPY2;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import org.junit.Test;

/** Tests that are common to {@code py_binary} and {@code py_test}. */
public abstract class PyExecutableConfiguredTargetTestBase extends PyBaseConfiguredTargetTestBase {

  private final String ruleName;

  protected PyExecutableConfiguredTargetTestBase(String ruleName) {
    super(ruleName);
    this.ruleName = ruleName;
  }

  /**
   * Returns the configured target with the given label while asserting that, if it is an executable
   * target, the executable is not produced by {@link FailAction}.
   *
   * <p>This serves as a drop-in replacement for {@link #getConfiguredTarget} that will also catch
   * unexpected deferred failures (e.g. {@code srcs_versions} validation failures) in {@code
   * py_binary} and {@code py_test} targets.
   */
  protected ConfiguredTarget getOkPyTarget(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    // It can be null without events due to b/26382502.
    Preconditions.checkNotNull(target, "target was null (is it misspelled or in error?)");
    Artifact executable = getExecutable(target);
    if (executable != null) {
      Action action = getGeneratingAction(executable);
      if (action instanceof FailAction) {
        throw new AssertionError(
            String.format(
                "execution of target would fail with error '%s'",
                ((FailAction) action).getErrorMessage()));
      }
    }
    return target;
  }

  /**
   * Gets the configured target for an executable Python rule (generally {@code py_binary} or {@code
   * py_test}) and asserts that it produces a deferred error via {@link FailAction}.
   *
   * @return the deferred error string
   */
  protected String getPyExecutableDeferredError(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    // It can be null without events due to b/26382502.
    Preconditions.checkNotNull(target, "target was null (is it misspelled or in error?)");
    Artifact executable = getExecutable(target);
    Preconditions.checkNotNull(
        executable, "executable was null (is this a py_binary/py_test target?)");
    Action action = getGeneratingAction(executable);
    assertThat(action).isInstanceOf(FailAction.class);
    return ((FailAction) action).getErrorMessage();
  }

  /** Asserts that a configured target has the given Python version. */
  protected void assertPythonVersionIs(String targetName, PythonVersion version) throws Exception {
    assertThat(getPythonVersion(getOkPyTarget(targetName))).isEqualTo(version);
  }

  /**
   * Sets the configuration, then asserts that a configured target has the given Python version.
   *
   * <p>The configuration is given as a series of "--flag=value" strings.
   */
  protected void assertPythonVersionIs_UnderNewConfig(
      String targetName, PythonVersion version, String... flags) throws Exception {
    useConfiguration(flags);
    assertPythonVersionIs(targetName, version);
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
          .that(getPythonVersion(getOkPyTarget(targetName)))
          .isEqualTo(version);
    }
  }

  private static String join(String... lines) {
    return String.join("\n", lines);
  }

  private String ruleDeclWithPyVersionAttr(String name, String version) {
    return join(
        ruleName + "(",
        "    name = '" + name + "',",
        "    srcs = ['" + name + ".py'],",
        "    python_version = '" + version + "'",
        ")");
  }

  @Test
  public void pyRuntimeInfoIsPresent() throws Exception {
    useConfiguration("--incompatible_use_python_toolchains=true");
    scratch.file(
        "pkg/BUILD", //
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        ")");
    assertThat(getConfiguredTarget("//pkg:foo").get(PyRuntimeInfo.PROVIDER)).isNotNull();
  }

  @Test
  public void versionAttr_UnknownValue() throws Exception {
    checkError(
        "pkg",
        "foo",
        // error:
        "invalid value in 'python_version' attribute: "
            + "has to be one of 'PY2' or 'PY3' instead of 'doesnotexist'",
        // build file:
        ruleDeclWithPyVersionAttr("foo", "doesnotexist"));
  }

  @Test
  public void versionAttr_BadValue() throws Exception {
    checkError(
        "pkg",
        "foo",
        // error:
        "invalid value in 'python_version' attribute: "
            + "has to be one of 'PY2' or 'PY3' instead of 'PY2AND3'",
        // build file:
        ruleDeclWithPyVersionAttr("foo", "PY2AND3"));
  }

  @Test
  public void versionAttr_GoodValue() throws Exception {
    scratch.file("pkg/BUILD", ruleDeclWithPyVersionAttr("foo", "PY2"));
    getOkPyTarget("//pkg:foo");
    assertNoEvents();
  }

  @Test
  public void py3IsDefaultFlag_SetsDefaultPythonVersion() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = ['foo.py'],",
        ")");
    assertPythonVersionIs_UnderNewConfig(
        "//pkg:foo",
        PythonVersion.PY2,
        "--incompatible_py3_is_default=false");
    assertPythonVersionIs_UnderNewConfig(
        "//pkg:foo",
        PythonVersion.PY3,
        "--incompatible_py3_is_default=true",
        // Keep the host Python as PY2, because we don't want to drag any implicit dependencies on
        // tools into PY3 for this test. (Doing so may require setting extra options to get it to
        // pass analysis.)
        "--host_force_python=PY2");
  }

  @Test
  public void py3IsDefaultFlag_DoesntOverrideExplicitVersion() throws Exception {
    scratch.file("pkg/BUILD", ruleDeclWithPyVersionAttr("foo", "PY2"));
    assertPythonVersionIs_UnderNewConfig(
        "//pkg:foo",
        PythonVersion.PY2,
        "--incompatible_py3_is_default=true",
        // Keep the host Python as PY2, because we don't want to drag any implicit dependencies on
        // tools into PY3 for this test. (Doing so may require setting extra options to get it to
        // pass analysis.)
        "--host_force_python=PY2");
  }

  @Test
  public void versionAttrWorks_WhenNotDefaultValue() throws Exception {
    assumesDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithPyVersionAttr("foo", "PY3"));

    assertPythonVersionIs("//pkg:foo", PythonVersion.PY3);
  }

  @Test
  public void versionAttrWorks_WhenSameAsDefaultValue() throws Exception {
    assumesDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithPyVersionAttr("foo", "PY2"));

    assertPythonVersionIs("//pkg:foo", PythonVersion.PY2);
  }

  @Test
  public void versionAttrTakesPrecedence_NonDefaultValue() throws Exception {
    assumesDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithPyVersionAttr("foo", "PY3"));

    assertPythonVersionIs_UnderNewConfig("//pkg:foo", PythonVersion.PY3, "--python_version=PY2");
  }

  @Test
  public void versionAttrTakesPrecedence_DefaultValue() throws Exception {
    assumesDefaultIsPY2();
    scratch.file("pkg/BUILD", ruleDeclWithPyVersionAttr("foo", "PY2"));

    assertPythonVersionIs_UnderNewConfig("//pkg:foo", PythonVersion.PY2, "--python_version=PY3");
  }

  @Test
  public void canBuildWithDifferentVersionAttrs() throws Exception {
    scratch.file(
        "pkg/BUILD",
        ruleDeclWithPyVersionAttr("foo_v2", "PY2"),
        ruleDeclWithPyVersionAttr("foo_v3", "PY3"));

    assertPythonVersionIs("//pkg:foo_v2", PythonVersion.PY2);
    assertPythonVersionIs("//pkg:foo_v3", PythonVersion.PY3);
  }

  @Test
  public void incompatibleSrcsVersion() throws Exception {
    reporter.removeHandler(failFastHandler); // We assert below that we don't fail at analysis.
    scratch.file(
        "pkg/BUILD",
        // build file:
        ruleName + "(",
        "    name = 'foo',",
        "    srcs = [':foo.py'],",
        "    srcs_version = 'PY2ONLY',",
        "    python_version = 'PY3')");
    assertThat(getPyExecutableDeferredError("//pkg:foo"))
        .contains("being built for Python 3 but (transitively) includes Python 2-only sources");
    // This is an execution-time error, not an analysis-time one. We fail by setting the generating
    // action to FailAction.
    assertNoEvents();
  }

  @Test
  public void targetInPackageWithHyphensOkIfSrcsFromOtherPackage() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        "exports_files(['foo.py'])");
    scratch.file(
        "pkg-with-hyphens/BUILD",
        ruleName + "(",
        "    name = 'foo',",
        "    main = '//pkg:foo.py',",
        "    srcs = ['//pkg:foo.py'])");
    getOkPyTarget("//pkg-with-hyphens:foo"); // should not fail
  }
}
