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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PythonOptions} and {@link PythonConfiguration}. */
@RunWith(JUnit4.class)
public class PythonConfigurationTest extends ConfigurationTestCase {

  // Do not mutate the returned PythonOptions - it will poison skyframe caches.
  private PythonOptions parsePythonOptions(String... cmdline) throws Exception {
    BuildConfigurationValue config = create(cmdline);
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
  public void getDefaultPythonVersion() throws Exception {
    PythonOptions withoutPy3IsDefaultOpts =
        parsePythonOptions("--incompatible_py3_is_default=false");
    PythonOptions withPy3IsDefaultOpts = parsePythonOptions("--incompatible_py3_is_default=true");
    assertThat(withoutPy3IsDefaultOpts.getDefaultPythonVersion()).isEqualTo(PythonVersion.PY2);
    assertThat(withPy3IsDefaultOpts.getDefaultPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getPythonVersion_FallBackOnDefaultPythonVersion() throws Exception {
    // Run it twice with two different values for the incompatible flag to confirm it's actually
    // reading getDefaultPythonVersion() and not some other source of default values.
    PythonOptions py2Opts = parsePythonOptions("--incompatible_py3_is_default=false");
    PythonOptions py3Opts = parsePythonOptions("--incompatible_py3_is_default=true");
    assertThat(py2Opts.getPythonVersion()).isEqualTo(PythonVersion.PY2);
    assertThat(py3Opts.getPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void canTransitionPythonVersion_Yes() throws Exception {
    PythonOptions opts = parsePythonOptions("--python_version=PY3");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY2)).isTrue();
  }

  @Test
  public void canTransitionPythonVersion_NoBecauseSameAsCurrent() throws Exception {
    PythonOptions opts = parsePythonOptions("--python_version=PY3");
    assertThat(opts.canTransitionPythonVersion(PythonVersion.PY3)).isFalse();
  }

  @Test
  public void setPythonVersion() throws Exception {
    PythonOptions opts = Options.parse(PythonOptions.class, "--python_version=PY2").getOptions();
    opts.setPythonVersion(PythonVersion.PY3);
    assertThat(opts.pythonVersion).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getExec_copiesMostValues() throws Exception {
    BuildOptions options =
        parseBuildOptions(
            /* starlarkOptions= */ ImmutableMap.of(),
            "--incompatible_py3_is_default=true",
            "--incompatible_py2_outputs_are_suffixed=true",
            "--build_python_zip=true",
            "--incompatible_use_python_toolchains=true");

    PythonOptions execOpts =
        AnalysisTestUtil.execOptions(options, skyframeExecutor, reporter).get(PythonOptions.class);

    assertThat(execOpts.incompatiblePy3IsDefault).isTrue();
    assertThat(execOpts.incompatiblePy2OutputsAreSuffixed).isTrue();
    assertThat(execOpts.buildPythonZip).isEqualTo(TriState.YES);
    assertThat(execOpts.incompatibleUsePythonToolchains).isTrue();
  }

  @Test
  public void getExec_appliesHostForcePython() throws Exception {
    assumesDefaultIsPY2();

    BuildOptions optsWithPythonVersionFlag =
        parseBuildOptions(
            /* starlarkOptions= */ ImmutableMap.of(),
            "--python_version=PY2",
            "--host_force_python=PY3");

    BuildOptions optsWithPy3IsDefaultFlag =
        parseBuildOptions(
            /* starlarkOptions= */ ImmutableMap.of(),
            "--incompatible_py3_is_default=true",
            // It's more interesting to set the incompatible flag true and force exec to PY2, than
            // it is to set the flag false and force exec to PY3.
            "--host_force_python=PY2");

    PythonOptions execOptsWithPythonVersionFlag =
        AnalysisTestUtil.execOptions(optsWithPythonVersionFlag, skyframeExecutor, reporter)
            .get(PythonOptions.class);
    PythonOptions execOptsWithPy3IsDefaultFlag =
        AnalysisTestUtil.execOptions(optsWithPy3IsDefaultFlag, skyframeExecutor, reporter)
            .get(PythonOptions.class);

    assertThat(execOptsWithPythonVersionFlag.getPythonVersion()).isEqualTo(PythonVersion.PY3);
    assertThat(execOptsWithPy3IsDefaultFlag.getPythonVersion()).isEqualTo(PythonVersion.PY2);
  }

  @Test
  public void getExec_py3IsDefaultFlagChangesExec() throws Exception {
    assumesDefaultIsPY2();
    BuildOptions options =
        parseBuildOptions(
            /* starlarkOptions= */ ImmutableMap.of(), "--incompatible_py3_is_default=true");

    PythonOptions execOpts =
        AnalysisTestUtil.execOptions(options, skyframeExecutor, reporter).get(PythonOptions.class);

    assertThat(execOpts.getPythonVersion()).isEqualTo(PythonVersion.PY3);
  }

  @Test
  public void getNormalized() throws Exception {
    PythonOptions opts = parsePythonOptions();
    PythonOptions normalizedOpts = (PythonOptions) opts.getNormalized();
    assertThat(normalizedOpts.pythonVersion).isEqualTo(PythonVersion.PY3);
  }
}
