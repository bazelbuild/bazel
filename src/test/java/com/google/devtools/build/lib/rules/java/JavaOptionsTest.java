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
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link JavaOptions}. */
@RunWith(JUnit4.class)
public class JavaOptionsTest extends BuildViewTestCase {
  @Test
  public void hostJavacOptions() throws Exception {
    BuildOptions options = targetConfig.getOptions().clone();
    options.get(JavaOptions.class).javacOpts = ImmutableList.of("-XDtarget");
    options.get(JavaOptions.class).hostJavacOpts = ImmutableList.of("-XDhost");

    BuildOptions execOptions = AnalysisTestUtil.execOptions(options, skyframeExecutor, reporter);
    assertThat(execOptions.get(JavaOptions.class).javacOpts).contains("-XDhost");
    assertThat(execOptions.get(JavaOptions.class).hostJavacOpts).contains("-XDhost");
  }
}
