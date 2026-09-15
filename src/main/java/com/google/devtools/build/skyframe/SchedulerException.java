// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import java.util.Set;
import javax.annotation.Nullable;

/** Wrapper exception that {@link Runnable}s can throw. */
class SchedulerException extends RuntimeException {
  private final SkyKey failedValue;
  private final ErrorInfo errorInfo;
  private final Set<SkyKey> rdepsToBubbleUpTo;

  private SchedulerException(
      @Nullable Exception cause,
      @Nullable ErrorInfo errorInfo,
      SkyKey failedValue,
      Set<SkyKey> rdepsToBubbleUpTo) {
    super(errorInfo != null ? errorInfo.getException() : cause);
    this.errorInfo = errorInfo;
    this.rdepsToBubbleUpTo = rdepsToBubbleUpTo;
    this.failedValue = Preconditions.checkNotNull(failedValue, errorInfo);
  }

  /**
   * Returns a SchedulerException wrapping an expected error, e.g. an error describing an expected
   * build failure when trying to evaluate the given value, that should cause Skyframe to produce
   * useful error information to the user.
   */
  static SchedulerException ofError(
      ErrorInfo errorInfo, SkyKey failedValue, Set<SkyKey> rdepsToBubbleUpTo) {
    Preconditions.checkNotNull(errorInfo);
    Preconditions.checkNotNull(rdepsToBubbleUpTo, "null rdeps: %s %s", errorInfo, failedValue);
    return new SchedulerException(
        errorInfo.getException(), errorInfo, failedValue, rdepsToBubbleUpTo);
  }

  /**
   * Returns a SchedulerException wrapping an InterruptedException, e.g. if the user interrupts
   * the build, that should cause Skyframe to exit as soon as possible.
   */
  static SchedulerException ofInterruption(InterruptedException cause, SkyKey failedValue) {
    return new SchedulerException(cause, null, failedValue, null);
  }

  SkyKey getFailedValue() {
    return failedValue;
  }

  @Nullable
  ErrorInfo getErrorInfo() {
    return errorInfo;
  }

  Set<SkyKey> getRdepsToBubbleUpTo() {
    return rdepsToBubbleUpTo;
  }
}
