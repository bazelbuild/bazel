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

import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;

/** Factory method for producing {@link Interrupted}-type {@link FailureDetail} messages. */
public class InterruptedFailureDetails {

  private InterruptedFailureDetails() {}

  /**
   * Returns a {@link DetailedExitCode} with {@link ExitCode#INTERRUPTED} and {@link FailureDetail}
   * having the provided message and {@link Interrupted.Code}.
   */
  public static DetailedExitCode detailedExitCode(String message, Interrupted.Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setInterrupted(Interrupted.newBuilder().setCode(detailedCode))
            .build());
  }

  /**
   * Returns an {@link AbruptExitException} with a {@link DetailedExitCode} from {@link
   * #detailedExitCode}.
   */
  public static AbruptExitException abruptExitException(
      String message, Interrupted.Code detailedCode) {
    return new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setInterrupted(Interrupted.newBuilder().setCode(detailedCode))
                .build()));
  }

  /**
   * Returns an {@link AbruptExitException} with a {@link DetailedExitCode} from {@link
   * #detailedExitCode} and the provided {@code cause}.
   */
  public static AbruptExitException abruptExitException(
      String message, Interrupted.Code detailedCode, Exception cause) {
    return new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setInterrupted(Interrupted.newBuilder().setCode(detailedCode))
                .build()),
        cause);
  }
}
