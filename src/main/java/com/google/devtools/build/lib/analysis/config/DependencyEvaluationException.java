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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.analysis.InconsistentAspectOrderException;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * Exception that signals an error during the evaluation of a configured target dependency.
 *
 * <p>If {@link DependencyEvaluationException#depReportedOwnError()}} is true, dependencies are
 * assumed to have reported their own errors. So if configured target P depends on configured target
 * D, and P fails because of a {@code DependencyEvaluationException} on D, P is responsible for
 * reporting its error details. P should only report what contextualizes P's relationship to D.
 *
 * <p>If {@link DependencyEvaluationException#depReportedOwnError()}} is false, P reports both
 * errors in consolidated form as it sees fit. For conceptual simplicity's sake, use this variation
 * sparingly.
 *
 * <p>The result is essentially an error reporting stack trace, but presented with user readability
 * in mind.
 */
public class DependencyEvaluationException extends Exception {
  /* Null denotes whatever default exit code callers choose. */
  @Nullable private final DetailedExitCode detailedExitCode;
  @Nullable private final Location location;
  private final boolean depReportedOwnError;

  private DependencyEvaluationException(
      Exception cause,
      @Nullable DetailedExitCode detailedExitCode,
      @Nullable Location location,
      boolean depReportedOwnError) {
    super(cause.getMessage(), cause);
    this.detailedExitCode = detailedExitCode;
    this.location = location;
    this.depReportedOwnError = depReportedOwnError;
  }

  public DependencyEvaluationException(
      ConfiguredValueCreationException cause, boolean depReportedOwnError) {
    this(cause, cause.getDetailedExitCode(), cause.getLocation(), depReportedOwnError);
  }

  public DependencyEvaluationException(InconsistentAspectOrderException cause) {
    // Calling logic doesn't provide an opportunity for this dependency to report its own error.
    // TODO(bazel-team): clean up the calling logic to eliminate this distinction.
    this(cause, /*detailedExitCode=*/ null, cause.getLocation(), /*depReportedOwnError=*/ false);
  }

  /** Returns the cause's {@link DetailedExitCode}. If null, the caller should choose a default. */
  @Nullable
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  @Nullable
  public Location getLocation() {
    return location;
  }

  public boolean depReportedOwnError() {
    return depReportedOwnError;
  }

  @Override
  public synchronized Exception getCause() {
    return (Exception) super.getCause();
  }
}
