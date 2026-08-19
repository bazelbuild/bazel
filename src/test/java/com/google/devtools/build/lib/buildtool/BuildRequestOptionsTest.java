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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link BuildRequestOptions}. */
@RunWith(JUnit4.class)
public class BuildRequestOptionsTest {

  private static BuildRequestOptions parse(String... args) throws OptionsParsingException {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildRequestOptions.class).build();
    parser.parse(args);
    return parser.getOptions(BuildRequestOptions.class);
  }

  @Test
  public void maxConcurrentActions_withoutAsyncExecution_isJobs() throws Exception {
    assertThat(
            parse("--jobs=17", "--experimental_async_execution_max_concurrent_actions=100")
                .getMaxConcurrentActions())
        .isEqualTo(17);
  }

  @Test
  public void maxConcurrentActions_withAsyncExecution_isMaxConcurrentActions() throws Exception {
    assertThat(
            parse(
                    "--jobs=17",
                    "--experimental_async_execution",
                    "--experimental_async_execution_max_concurrent_actions=100")
                .getMaxConcurrentActions())
        .isEqualTo(100);
  }

  @Test
  public void maxConcurrentActions_belowJobs_isClampedToJobs() throws Exception {
    assertThat(
            parse(
                    "--jobs=17",
                    "--experimental_async_execution",
                    "--experimental_async_execution_max_concurrent_actions=0")
                .getMaxConcurrentActions())
        .isEqualTo(17);
  }

  @Test
  public void maxConcurrentActions_aboveMaxJobs_isClampedToMaxJobs() throws Exception {
    assertThat(
            parse(
                    "--jobs=17",
                    "--experimental_async_execution",
                    "--experimental_async_execution_max_concurrent_actions="
                        + (BuildRequestOptions.MAX_JOBS + 1))
                .getMaxConcurrentActions())
        .isEqualTo(BuildRequestOptions.MAX_JOBS);
  }
}
