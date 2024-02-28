// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.rewinding;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.server.FailureDetails.ActionRewinding;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;

/** Exception thrown by {@link ActionRewindStrategy} when it cannot compute a rewind plan. */
public final class ActionRewindException extends Exception implements DetailedException {
  private final ActionRewinding.Code code;

  ActionRewindException(String message, ActionRewinding.Code code) {
    super(message);
    this.code = checkNotNull(code);
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(getMessage())
            .setActionRewinding(ActionRewinding.newBuilder().setCode(code))
            .build());
  }
}
