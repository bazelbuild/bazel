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
import static com.google.devtools.build.lib.rules.python.PythonTestUtils.assumesDefaultIsPY2;
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

  private PythonOptions parsePythonOptions(String... cmdline) throws Exception {
    BuildConfiguration config = create(cmdline);
    return config.getOptions().get(PythonOptions.class);
  }

  @Test
  public void invalidTargetPythonValue_NotATargetValue() {
    OptionsParsingException expected =
        assertThrows(OptionsParsingException.class, () -> create("--python_version=PY2AND3"));
    assertThat(expected).hasMessageThat().contains("Not a valid Python major version");
  }

  @Test
  public void invalidTargetPythonValue_UnknownValue() {
    OptionsParsingException expected =
        assertThrows(OptionsParsingException.class, () -> create("--python_version=BEETLEJUICE"));
    assertThat(expected).hasMessageThat().contains("Not a valid Python major version");
  }

  @Test
  public void py3IsDefaultFlagRequiresNewSemanticsFlag() throws Exception {
    checkError(
        "cannot enable `--incompatible_py3_is_default` without also enabling "
            + "`--incompatible_allow_python_version_transitions`",
        "--incompatible_allow_python_version_transitions=false",
        "--incompatible_py3_is_default=true");
  }

  @Test
  public void getDefaultPythonVersion() throws Exception {
    // --incompatible_py3_is_default requires --incompatible_allow_python_version_transitions
    PythonOptions withoutPy3IsDefaultOpts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_py3_is_default=false");
    PythonOptions withPy3IsDefaultOpts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_py3_is_default=true");
    assertThat(withoutPy3IsDefaultOpts.getDefaultPythonVersion()).isEqualTo(PythonVersion.PY2);
    assertThat(withPy3IsDefaultOpts.getDefaultPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getPythonVersion_FallBackOnDefaultPythonVersion() throws Exception {
    // Run it twice with two different values for the incompatible flag to confirm it's actually
    // reading getDefaultPythonVersion() and not some other source of default values. Note that
    // --incompatible_py3_is_default requires --incompatible_allow_python_version_transitions.
    PythonOptions py2Opts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_py3_is_default=false");
    PythonOptions py3Opts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_py3_is_default=true");
    assertThat(py2Opts.getPythonVersion()).isEqualTo(PythonVersion.PY2);
    assertThat(py3Opts.getPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void canTransitionPythonVersion_OldSemantics_Yes() throws Exception {
    assumesDefaultIsPY2();
    PythonOptions opts =
        parsePythonOptions("--incompatible_allow_python_version_transitions=false");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY3)).isTrue();
  }

  @Test
  public void canTransitionPythonVersion_OldSemantics_NoBecauseAlreadySet() throws Exception {
    assumesDefaultIsPY2();
    PythonOptions opts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=false", "--python_version=PY2");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY3)).isFalse();
  }

  @Test
  public void canTransitionPythonVersion_OldSemantics_NoBecauseNewValueSameAsDefault()
      throws Exception {
    assumesDefaultIsPY2();
    PythonOptions opts =
        parsePythonOptions("--incompatible_allow_python_version_transitions=false");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY2)).isFalse();
  }

  @Test
  public void canTransitionPythonVersion_NewSemantics_Yes() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true", "--python_version=PY3");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY2)).isTrue();
  }

  @Test
  public void canTransitionPythonVersion_NewSemantics_NoBecauseSameAsCurrent() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_remove_old_python_version_api=false",
            "--python_version=PY3");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY3)).isFalse();
  }

  @Test
  public void setPythonVersion_OldApiEnabled() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--incompatible_remove_old_python_version_api=false",
            "--python_version=PY2");
    opts.setPythonVersion(PythonVersion.PY3);
    assertThat(opts.pythonVersion).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void setPythonVersion_OldApiDisabled() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--incompatible_remove_old_python_version_api=true", "--python_version=PY2");
    opts.setPythonVersion(PythonVersion.PY3);
    assertThat(opts.pythonVersion).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getHost_CopiesMostValues() throws Exception {
    PythonOptions opts =
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_remove_old_python_version_api=true",
            "--incompatible_py3_is_default=true",
            "--incompatible_py2_outputs_are_suffixed=true",
            "--build_python_zip=true",
            "--incompatible_disallow_legacy_py_provider=true",
            "--incompatible_use_python_toolchains=true");
    PythonOptions hostOpts = (PythonOptions) opts.getHost();
    assertThat(hostOpts.incompatibleAllowPythonVersionTransitions).isTrue();
    assertThat(hostOpts.incompatibleRemoveOldPythonVersionApi).isTrue();
    assertThat(hostOpts.incompatiblePy3IsDefault).isTrue();
    assertThat(hostOpts.incompatiblePy2OutputsAreSuffixed).isTrue();
    assertThat(hostOpts.buildPythonZip).isEqualTo(TriState.YES);
    assertThat(hostOpts.incompatibleDisallowLegacyPyProvider).isTrue();
    assertThat(hostOpts.incompatibleUsePythonToolchains).isTrue();
  }

  @Test
  public void getHost_AppliesHostForcePython() throws Exception {
    assumesDefaultIsPY2();
    PythonOptions optsWithPythonVersionFlag =
        parsePythonOptions("--python_version=PY2", "--host_force_python=PY3");
    PythonOptions optsWithPy3IsDefaultFlag =
        // --incompatible_py3_is_default requires --incompatible_allow_python_version_transitions
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_py3_is_default=true",
            // It's more interesting to set the incompatible flag true and force host to PY2, than
            // it is to set the flag false and force host to PY3.
            "--host_force_python=PY2");
    PythonOptions hostOptsWithPythonVersionFlag =
        (PythonOptions) optsWithPythonVersionFlag.getHost();
    PythonOptions hostOptsWithPy3IsDefaultFlag = (PythonOptions) optsWithPy3IsDefaultFlag.getHost();
    assertThat(hostOptsWithPythonVersionFlag.getPythonVersion()).isEqualTo(PythonVersion.PY3);
    assertThat(hostOptsWithPy3IsDefaultFlag.getPythonVersion()).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void getHost_Py3IsDefaultFlagChangesHost() throws Exception {
    assumesDefaultIsPY2();
    PythonOptions opts =
        // --incompatible_py3_is_default requires --incompatible_allow_python_version_transitions
        parsePythonOptions(
            "--incompatible_allow_python_version_transitions=true",
            "--incompatible_py3_is_default=true");
    PythonOptions hostOpts = (PythonOptions) opts.getHost();
    assertThat(hostOpts.getPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getNormalized_OldSemantics() throws Exception {
    PythonOptions opts =
        parsePythonOptions("--incompatible_allow_python_version_transitions=false");
    PythonOptions normalizedOpts = (PythonOptions) opts.getNormalized();
    assertThat(normalizedOpts.pythonVersion).isNull();
  }

  @Test
  public void getNormalized_NewSemantics() throws Exception {
    assumesDefaultIsPY2();
    PythonOptions opts = parsePythonOptions("--incompatible_allow_python_version_transitions=true");
    PythonOptions normalizedOpts = (PythonOptions) opts.getNormalized();
    assertThat(normalizedOpts.pythonVersion).isEqualTo(PythonVersion.PY2);
  }
}
