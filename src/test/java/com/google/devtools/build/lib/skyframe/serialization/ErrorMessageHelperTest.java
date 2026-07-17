// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.serialization.ErrorMessageHelper.MAX_ERRORS_TO_REPORT;
import static com.google.devtools.build.lib.skyframe.serialization.ErrorMessageHelper.getErrorMessage;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ErrorMessageHelperTest {

  @Test
  public void getErrorMessage_noExceptions() {
    String message = getErrorMessage(ImmutableList.of());
    assertThat(message).isEmpty();
  }

  @Test
  public void getErrorMessage_singleException() {
    String message = getErrorMessage(ImmutableList.of(new RuntimeException("Test exception")));
    assertThat(message).contains("Test exception");
  }

  @Test
  public void getErrorMessage_multipleExceptions() {
    String message =
        getErrorMessage(
            ImmutableList.of(
                new RuntimeException("Test exception 1"),
                new RuntimeException("Test exception 2")));
    assertThat(message).contains("Test exception 1");
    assertThat(message).contains("Test exception 2");
  }

  @Test
  public void getErrorMessage_exactLimitExceptions() {
    ImmutableList.Builder<Throwable> exceptions = ImmutableList.builder();
    for (int i = 0; i < MAX_ERRORS_TO_REPORT; i++) {
      exceptions.add(new RuntimeException("Error " + i));
    }
    String message = getErrorMessage(exceptions.build());
    assertThat(message).contains("There were 5 write errors.");
    assertThat(message).doesNotContain("Only the first");
    for (int i = 0; i < MAX_ERRORS_TO_REPORT; i++) {
      assertThat(message).contains("Error " + i);
    }
  }

  @Test
  public void getErrorMessage_moreThanLimitExceptions() {
    ImmutableList.Builder<Throwable> exceptions = ImmutableList.builder();
    for (int i = 0; i < 6; i++) {
      exceptions.add(new RuntimeException("Error " + i));
    }
    String message = getErrorMessage(exceptions.build());
    assertThat(message).contains("There were 6 write errors. Only the first 5 will be reported.");
    for (int i = 0; i < MAX_ERRORS_TO_REPORT; i++) {
      assertThat(message).contains("Error " + i);
    }
    assertThat(message).doesNotContain("Error 5");
  }
}
