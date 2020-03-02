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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted.InterruptedCode;
import com.google.protobuf.ProtocolMessageEnum;

/** Utilities for constructing and managing {@link FailureDetail}s. */
public class FailureDetailUtil {

  private FailureDetailUtil() {}

  /**
   * Returns a {@link FailureDetail} specifying {@code code} in its {@link Interrupted} submessage.
   */
  public static FailureDetail interrupted(InterruptedCode code) {
    return FailureDetail.newBuilder()
        .setInterrupted(Interrupted.newBuilder().setCode(code))
        .build();
  }

  /** Returns the registered {@link ExitCode} associated with a {@link FailureDetail} message. */
  public static ExitCode getExitCode(FailureDetail failureDetail) {
    // TODO(mschaller): Consider specializing for unregistered exit codes here, if absolutely
    //  necessary.
    int numericExitCode = FailureDetailUtil.getNumericExitCode(failureDetail);
    return Preconditions.checkNotNull(
        ExitCode.forCode(numericExitCode), "No ExitCode for numericExitCode %s", numericExitCode);
  }

  /** Returns the numeric exit code associated with a {@link FailureDetail} message. */
  private static int getNumericExitCode(FailureDetail failureDetail) {
    // TODO(mschaller): generalize this for arbitrary FailureDetail messages.
    Preconditions.checkArgument(failureDetail.hasInterrupted());
    return getNumericExitCode(failureDetail.getInterrupted().getCode());
  }

  /**
   * Returns the numeric exit code associated with a {@link FailureDetail} submessage's subcategory
   * enum value.
   */
  private static int getNumericExitCode(ProtocolMessageEnum code) {
    Preconditions.checkArgument(
        code.getValueDescriptor().getOptions().hasExtension(FailureDetails.metadata),
        "Enum value %s has no FailureDetails.metadata",
        code);
    return code.getValueDescriptor()
        .getOptions()
        .getExtension(FailureDetails.metadata)
        .getExitCode();
  }
}
