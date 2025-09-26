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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.common.options.TriState;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PythonOptions} and {@link PythonConfiguration}. */
@RunWith(JUnit4.class)
public class PythonConfigurationTest extends ConfigurationTestCase {

  @Test
  public void getExec_copiesMostValues() throws Exception {
    BuildOptions options =
        parseBuildOptions(
            /* starlarkOptions= */ ImmutableMap.of(),
            "--build_python_zip=true",
            "--experimental_py_binaries_include_label=true");

    PythonOptions execOpts =
        AnalysisTestUtil.execOptions(options, skyframeExecutor, reporter).get(PythonOptions.class);

    assertThat(execOpts.buildPythonZip).isEqualTo(TriState.YES);
    assertThat(execOpts.includeLabelInPyBinariesLinkstamp).isTrue();
  }
}
