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

import com.google.devtools.build.lib.server.FailureDetails.ActionRewinding;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;

/** Exception thrown by {@link ActionRewindStrategy} when it cannot compute a rewind plan. */
public abstract sealed class ActionRewindException extends Exception implements DetailedException {

  ActionRewindException(String message) {
    super(message);
  }

  static final class GenericActionRewindException extends ActionRewindException {
    private final ActionRewinding.Code code;

    GenericActionRewindException(String message, ActionRewinding.Code code) {
      super(message);
      this.code = code;
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

  static final class FallbackToBuildRewindingException extends ActionRewindException {
    FallbackToBuildRewindingException(String message) {
      super(message);
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return DetailedExitCode.of(
          FailureDetail.newBuilder()
              .setMessage(getMessage())
              .setSpawn(Spawn.newBuilder().setCode(Spawn.Code.REMOTE_CACHE_EVICTED))
              .build());
    }
  }
}
