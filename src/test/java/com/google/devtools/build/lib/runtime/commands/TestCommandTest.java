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

package com.google.devtools.build.lib.runtime.commands;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.common.options.OptionsParser;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TestCommand}. */
@RunWith(JUnit4.class)
public final class TestCommandTest {

  @Test
  public void editOptions_streamedOutput_forcesExclusiveStrategy() throws Exception {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(ExecutionOptions.class, CoreOptions.class, TestOptions.class)
            .build();
    parser.parse("--test_output=streamed", "--test_strategy=standalone");

    new TestCommand().editOptions(parser);

    ExecutionOptions executionOptions = parser.getOptions(ExecutionOptions.class);
    assertThat(executionOptions.getTestStrategy()).isEqualTo("exclusive");
  }

  @Test
  public void editOptions_nonStreamedOutput_preservesTestStrategy() throws Exception {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(ExecutionOptions.class, CoreOptions.class, TestOptions.class)
            .build();
    parser.parse("--test_output=errors", "--test_strategy=standalone");

    new TestCommand().editOptions(parser);

    ExecutionOptions executionOptions = parser.getOptions(ExecutionOptions.class);
    assertThat(executionOptions.getTestStrategy()).isEqualTo("standalone");
  }

  @Test
  public void streamedOutputWarning_containsGuidanceForParallelExecution() {
    assertThat(TestCommand.STREAMED_OUTPUT_WARNING).contains("exclusive");
    assertThat(TestCommand.STREAMED_OUTPUT_WARNING).contains("parallel");
    assertThat(TestCommand.STREAMED_OUTPUT_WARNING).contains("--test_output=errors");
    assertThat(TestCommand.STREAMED_OUTPUT_WARNING).contains("--test_output=all");
  }
}
