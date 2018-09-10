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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.analysis.util.ConfigurationTestCase;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PythonConfiguration}. */
@RunWith(JUnit4.class)
public class PythonConfigurationTest extends ConfigurationTestCase {

  @Test
  public void invalidForcePythonValue_NotATargetValue() throws Exception {
    checkError("'--force_python' argument must be 'PY2' or 'PY3'", "--force_python=PY2AND3");
  }

  @Test
  public void invalidForcePythonValue_UnknownValue() {
    OptionsParsingException expected =
        assertThrows(
            OptionsParsingException.class,
            () -> create("--force_python=BEETLEJUICE"));
    assertThat(expected).hasMessageThat()
        .contains("While parsing option --force_python=BEETLEJUICE: Not a valid Python version");
  }
}
