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

package com.google.devtools.build.lib.buildeventservice;

import com.google.devtools.build.lib.server.FailureDetails.BuildProgress;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.ExitCode;
import javax.annotation.Nullable;

/** Event fired from {@link BuildEventServiceUploader}. */
public class BuildEventServiceAvailabilityEvent {
  private final ExitCode exitCode;
  @Nullable private final FailureDetail failureDetail;

  public BuildEventServiceAvailabilityEvent(
      ExitCode exitCode, @Nullable FailureDetail failureDetail) {
    this.exitCode = exitCode;
    this.failureDetail = failureDetail;
  }

  public static BuildEventServiceAvailabilityEvent ofSuccess() {
    return new BuildEventServiceAvailabilityEvent(ExitCode.SUCCESS, null);
  }

  /**
   * Returns {@link ExitCode.SUCCESS} if the build event upload was a success, otherwise, return an
   * exit code that corresponds to the error that occurred during the build event upload.
   */
  public ExitCode getExitCode() {
    return exitCode;
  }

  /**
   * Returns a failure detail containing the status of the build event that was uploaded to the
   * build event service. This returns null if the upload completed successfully, otherwise, the
   * contents will contain an {@link ExitCode} and a {@link BuildProgress.Code}.
   */
  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }
}
