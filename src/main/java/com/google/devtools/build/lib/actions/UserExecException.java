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

import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;

/**
 * An ExecException that is related to the failure of an Action and therefore very likely the user's
 * fault.
 */
public class UserExecException extends ExecException {

  private final FailureDetail failureDetail;

  public UserExecException(FailureDetail failureDetail) {
    super(failureDetail.getMessage());
    this.failureDetail = failureDetail;
  }

  public UserExecException(Throwable cause, FailureDetail failureDetail) {
    super(failureDetail.getMessage(), cause);
    this.failureDetail = failureDetail;
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return failureDetail.toBuilder().setMessage(message).build();
  }
}
