// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import java.io.IOException;

/**
 * An ExecException which reports an issue executing an action due to an external problem on the
 * local system.
 *
 * <p>The most common use of this exception is to wrap an "unexpected" {@link IOException} thrown by
 * an lower-level file system access or local process execution, e.g., failure to create a temporary
 * directory or denied file system access.
 */
public class EnvironmentalExecException extends ExecException {
  private final FailureDetail failureDetail;

  public EnvironmentalExecException(IOException cause, FailureDetails.Execution.Code code) {
    super(cause);
    this.failureDetail =
        FailureDetail.newBuilder().setExecution(Execution.newBuilder().setCode(code)).build();
  }

  public EnvironmentalExecException(Throwable cause, FailureDetail failureDetail) {
    super(failureDetail.getMessage(), cause);
    this.failureDetail = failureDetail;
  }

  public EnvironmentalExecException(FailureDetail failureDetail) {
    super(failureDetail.getMessage());
    this.failureDetail = failureDetail;
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return failureDetail.toBuilder().setMessage(message).build();
  }
}
