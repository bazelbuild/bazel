// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.OptionsParser;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link JavaOptions}.
 */
@RunWith(JUnit4.class)
public class JavaOptionsTest {
  @Test
  public void hostJavacOptions() throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(CoreOptions.class, JavaOptions.class).build();
    parser.parse("--javacopt=-XDtarget", "--host_javacopt=-XDhost");
    BuildOptions buildOptions =
        BuildOptions.of(
            ImmutableList.<Class<? extends FragmentOptions>>of(
                CoreOptions.class, JavaOptions.class),
            parser);

    assertThat(buildOptions.get(JavaOptions.class).javacOpts).contains("-XDtarget");
    assertThat(buildOptions.get(JavaOptions.class).hostJavacOpts).contains("-XDhost");
    assertThat(((JavaOptions) buildOptions.get(JavaOptions.class).getExec()).javacOpts)
        .contains("-XDhost");
  }
}
