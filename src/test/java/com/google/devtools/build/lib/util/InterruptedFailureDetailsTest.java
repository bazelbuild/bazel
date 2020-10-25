// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted.Code;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link InterruptedFailureDetails}. */
@RunWith(JUnit4.class)
public class InterruptedFailureDetailsTest {

  @Test
  public void detailedExitCode() {
    DetailedExitCode detailedExitCode =
        InterruptedFailureDetails.detailedExitCode("myMessage", Code.BUILD);
    assertThat(detailedExitCode)
        .isEqualTo(
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage("myMessage")
                    .setInterrupted(Interrupted.newBuilder().setCode(Code.BUILD))
                    .build()));
  }

  @Test
  public void abruptExitException() {
    AbruptExitException abruptExitException =
        InterruptedFailureDetails.abruptExitException("myMessage", Code.BUILD);
    assertThat(abruptExitException).hasMessageThat().isEqualTo("myMessage");
    assertThat(abruptExitException.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(abruptExitException.getDetailedExitCode())
        .isEqualTo(InterruptedFailureDetails.detailedExitCode("myMessage", Code.BUILD));
  }

  @Test
  public void abruptExitExceptionWithCause() {
    Exception cause = new Exception();
    AbruptExitException abruptExitException =
        InterruptedFailureDetails.abruptExitException("myMessage", Code.BUILD, cause);
    assertThat(abruptExitException).hasMessageThat().isEqualTo("myMessage");
    assertThat(abruptExitException).hasCauseThat().isSameInstanceAs(cause);
    assertThat(abruptExitException.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(abruptExitException.getDetailedExitCode())
        .isEqualTo(InterruptedFailureDetails.detailedExitCode("myMessage", Code.BUILD));
  }
}
