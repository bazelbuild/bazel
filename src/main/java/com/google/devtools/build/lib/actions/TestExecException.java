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
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TestAction;

/** An TestExecException that is related to the failure of a TestAction. */
public final class TestExecException extends ExecException {
  private final FailureDetails.TestAction.Code detailedCode;

  public TestExecException(String message, FailureDetails.TestAction.Code detailedCode) {
    super(message);
    this.detailedCode = detailedCode;
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return FailureDetail.newBuilder()
        .setTestAction(TestAction.newBuilder().setCode(detailedCode))
        .setMessage(message)
        .build();
  }
}
