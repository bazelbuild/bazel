// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.common.options.OptionsParser;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Regression tests for {@link HostJvmStartupOptions}. */
@RunWith(JUnit4.class)
public class HostJvmStartupOptionsTest {

  @Test
  public void useSystemCertsIsFalseByDefault() throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(HostJvmStartupOptions.class).build();
    parser.parse();
    HostJvmStartupOptions result = parser.getOptions(HostJvmStartupOptions.class);
    assertThat(result.getUseSystemCerts()).isFalse();
  }

  @Test
  public void useSystemCertsCanBeEnabled() throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(HostJvmStartupOptions.class).build();
    parser.parse("--use_system_certs");
    HostJvmStartupOptions result = parser.getOptions(HostJvmStartupOptions.class);
    assertThat(result.getUseSystemCerts()).isTrue();
  }
}
