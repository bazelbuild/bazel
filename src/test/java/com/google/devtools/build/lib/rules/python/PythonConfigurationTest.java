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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PythonOptions} and {@link PythonConfiguration}. */
@RunWith(JUnit4.class)
public class PythonConfigurationTest extends ConfigurationTestCase {

  private void ensureDefaultIsPY2() {
    // Ensure that the expected value differs from the default value, so that if the code under test
    // ever returns the default value where it shouldn't have, the test doesn't spuriously succeed.
    assertWithMessage(
            "This test case is written with the assumption that the default is Python 2. When "
                + "updating the default to Python 3, flip all the PY2/PY3 constants in the test "
                + "case and this helper.")
        .that(PythonVersion.DEFAULT_TARGET_VALUE)
        .isEqualTo(PythonVersion.PY2);
  }

  private PythonOptions parsePythonOptions(String... cmdline) throws Exception {
    BuildConfiguration config = create(cmdline);
    return config.getOptions().get(PythonOptions.class);
  }

  @Test
  public void invalidTargetPythonValue_NotATargetValue() {
    OptionsParsingException expected =
        assertThrows(OptionsParsingException.class, () -> create("--force_python=PY2AND3"));
    assertThat(expected).hasMessageThat().contains("Not a valid Python major version");
  }

  @Test
  public void invalidTargetPythonValue_UnknownValue() {
    OptionsParsingException expected =
        assertThrows(
            OptionsParsingException.class,
            () -> create("--force_python=BEETLEJUICE"));
    assertThat(expected).hasMessageThat().contains("Not a valid Python major version");
  }

  @Test
  public void getPythonVersion_OldApi_NewFlagIgnored() throws Exception {
    ensureDefaultIsPY2();
    // --python_version should be ignored by getPythonVersion (and in fact would be disallowed in
    // the analysis phase).
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=false",
            "--force_python=PY3",
            "--python_version=PY2");
    assertThat(opts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getPythonVersion_OldApi_HardcodedDefaultWhenOmitted() throws Exception {
    ensureDefaultIsPY2();
    PythonOptions opts = parsePythonOptions("--experimental_better_python_version_mixing=false");
    assertThat(opts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void getPythonVersion_NewApi_NewFlagTakesPrecedence() throws Exception {
    ensureDefaultIsPY2();
    // --force_python is superseded by --python_version.
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=true",
            "--force_python=PY2",
            "--python_version=PY3");
    assertThat(opts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getPythonVersion_NewApi_FallBackOnOldFlag() throws Exception {
    ensureDefaultIsPY2();
    // --force_python is used because --python_version is absent.
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=true", "--force_python=PY3");
    assertThat(opts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getPythonVersion_NewApi_HardcodedDefaultWhenOmitted() throws Exception {
    ensureDefaultIsPY2();
    PythonOptions opts = parsePythonOptions("--experimental_better_python_version_mixing=true");
    assertThat(opts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void canTransitionPythonVersion_OldApi_Yes() throws Exception {
    ensureDefaultIsPY2();
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=false",
            // --python_version should be ignored.
            "--python_version=PY2");
    boolean result = opts.canTransitionPythonVersion(PythonVersion.PY3);
    assertThat(result).isTrue();
  }

  @Test
  public void canTransitionPythonVersion_OldApi_NoBecauseAlreadySet() throws Exception {
    ensureDefaultIsPY2();
    // --force_python can't be changed once explicitly set.
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=false", "--force_python=PY2");
    boolean result = opts.canTransitionPythonVersion(PythonVersion.PY3);
    assertThat(result).isFalse();
  }

  @Test
  public void canTransitionPythonVersion_OldApi_NoBecauseNewValueSameAsDefault() throws Exception {
    ensureDefaultIsPY2();
    // --force_python can't be changed to the hard-coded default value.
    PythonOptions opts = parsePythonOptions("--experimental_better_python_version_mixing=false");
    boolean result = opts.canTransitionPythonVersion(PythonVersion.PY2);
    assertThat(result).isFalse();
  }

  @Test
  public void canTransitionPythonVersion_NewApi_Yes() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=true", "--python_version=PY3");
    boolean result = opts.canTransitionPythonVersion(PythonVersion.PY2);
    assertThat(result).isTrue();
  }

  @Test
  public void canTransitionPythonVersion_NewApi_No() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=true",
            // Omit --python_version to confirm that we're still seen as already in PY3 mode due to
            // --force_python.
            "--force_python=PY3");
    boolean result = opts.canTransitionPythonVersion(PythonVersion.PY3);
    assertThat(result).isFalse();
  }

  @Test
  public void setPythonVersion_OldApi() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=false",
            "--force_python=PY2",
            // --python_version should be ignored.
            "--python_version=PY2");
    opts.setPythonVersion(PythonVersion.PY3);
    assertThat(opts.experimentalBetterPythonVersionMixing).isFalse();
    assertThat(opts.forcePython).isEqualTo(PythonVersion.PY3);
    assertThat(opts.pythonVersion).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void setPythonVersion_NewApi() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=true",
            // --force_python should be ignored.
            "--force_python=PY2",
            "--python_version=PY2");
    opts.setPythonVersion(PythonVersion.PY3);
    assertThat(opts.experimentalBetterPythonVersionMixing).isTrue();
    assertThat(opts.forcePython).isEqualTo(PythonVersion.PY2);
    assertThat(opts.pythonVersion).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getHost_OldSemantics() throws Exception {
    ensureDefaultIsPY2();
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=false",
            "--force_python=PY2",
            "--host_force_python=PY3",
            "--build_python_zip=true");
    PythonOptions newOpts = (PythonOptions) opts.getHost();
    assertThat(newOpts.experimentalBetterPythonVersionMixing).isFalse();
    assertThat(newOpts.forcePython).isEqualTo(PythonVersion.PY3);
    assertThat(newOpts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY3);
    assertThat(newOpts.buildPythonZip).isEqualTo(TriState.YES);
  }

  @Test
  public void getHost_NewSemantics() throws Exception {
    ensureDefaultIsPY2();
    PythonOptions opts =
        parsePythonOptions(
            "--experimental_better_python_version_mixing=true",
            "--python_version=PY2",
            "--host_force_python=PY3",
            "--build_python_zip=true");
    PythonOptions newOpts = (PythonOptions) opts.getHost();
    assertThat(newOpts.experimentalBetterPythonVersionMixing).isTrue();
    assertThat(newOpts.pythonVersion).isEqualTo(PythonVersion.PY3);
    assertThat(newOpts.NEWgetPythonVersion()).isEqualTo(PythonVersion.PY3);
    assertThat(newOpts.buildPythonZip).isEqualTo(TriState.YES);
  }
}
