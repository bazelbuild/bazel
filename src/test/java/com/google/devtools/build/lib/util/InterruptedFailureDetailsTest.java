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
    DetailedExitCode detailedExitCode = InterruptedFailureDetails.detailedExitCode("myMessage");
    assertThat(detailedExitCode)
        .isEqualTo(
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage("myMessage")
                    .setInterrupted(Interrupted.newBuilder().setCode(Code.INTERRUPTED))
                    .build()));
  }

  @Test
  public void abruptExitException() {
    AbruptExitException abruptExitException =
        InterruptedFailureDetails.abruptExitException("myMessage");
    assertThat(abruptExitException).hasMessageThat().isEqualTo("myMessage");
    assertThat(abruptExitException.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(abruptExitException.getDetailedExitCode())
        .isEqualTo(InterruptedFailureDetails.detailedExitCode("myMessage"));
  }

  @Test
  public void abruptExitExceptionWithCause() {
    Exception cause = new Exception();
    AbruptExitException abruptExitException =
        InterruptedFailureDetails.abruptExitException("myMessage", cause);
    assertThat(abruptExitException).hasMessageThat().isEqualTo("myMessage");
    assertThat(abruptExitException).hasCauseThat().isSameInstanceAs(cause);
    assertThat(abruptExitException.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(abruptExitException.getDetailedExitCode())
        .isEqualTo(InterruptedFailureDetails.detailedExitCode("myMessage"));
  }

  @Test
  public void detailedExitCode_nullMessage_returnsInterruptedExitCode() {
    DetailedExitCode detailedExitCode = InterruptedFailureDetails.detailedExitCode(null);
    assertThat(detailedExitCode.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(detailedExitCode.getFailureDetail().getMessage()).isEqualTo("interrupted");
    assertThat(detailedExitCode.getFailureDetail().getInterrupted().getCode())
        .isEqualTo(Code.INTERRUPTED);
  }

  @Test
  public void abruptExitException_nullMessage_returnsInterruptedExitCode() {
    AbruptExitException abruptExitException = InterruptedFailureDetails.abruptExitException(null);
    assertThat(abruptExitException).hasMessageThat().isEqualTo("interrupted");
    assertThat(abruptExitException.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(abruptExitException.getDetailedExitCode().getFailureDetail().getMessage())
        .isEqualTo("interrupted");
  }

  @Test
  public void abruptExitExceptionWithCause_nullMessage_returnsInterruptedExitCode() {
    Exception cause = new Exception();
    AbruptExitException abruptExitException =
        InterruptedFailureDetails.abruptExitException(null, cause);
    assertThat(abruptExitException).hasMessageThat().isEqualTo("interrupted");
    assertThat(abruptExitException).hasCauseThat().isSameInstanceAs(cause);
    assertThat(abruptExitException.getExitCode()).isEqualTo(ExitCode.INTERRUPTED);
    assertThat(abruptExitException.getDetailedExitCode().getFailureDetail().getMessage())
        .isEqualTo("interrupted");
  }
}
