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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/**
 * Exception indicating an attempt to access a target which is not found or does
 * not exist.
 */
public class NoSuchTargetException extends NoSuchThingException {
  /**
   * This factory is used when {@link Target} was loaded but isn't available to the caller.
   *
   * <p>This is used when an error is bubbled up from a child to parent {@link
   * com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction} invocation.
   */
  public static NoSuchTargetException createForParentPropagation(Label label) {
    return new NoSuchTargetException(label);
  }

  @Nullable private final Label label;
  private final boolean hasTarget;

  public NoSuchTargetException(String message) {
    this(
        message,
        /*label=*/ null,
        /*hasTarget=*/ false);
  }

  public NoSuchTargetException(Label label, String message) {
    this(
        "no such target '" + label + "': " + message,
        label,
        /*hasTarget=*/ false);
  }

  public NoSuchTargetException(Target targetInError) {
    this(targetInError.getLabel());
  }

  private NoSuchTargetException(Label label) {
    this(
        "Target '" + label + "' contains an error and its package is in error",
        label,
        /* hasTarget= */ true);
  }

  private NoSuchTargetException(String message, @Nullable Label label, boolean hasTarget) {
    // TODO(bazel-team): Does the exception matter?
    super(
        message,
        hasTarget ? new BuildFileContainsErrorsException(label.getPackageIdentifier()) : null);
    this.label = label;
    this.hasTarget = hasTarget;
  }

  @Nullable
  public Label getLabel() {
    return label;
  }

  /** Return whether parsing completed enough to construct the target. */
  public boolean hasTarget() {
    return hasTarget;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    DetailedExitCode uncheckedDetailedExitCode = getUncheckedDetailedExitCode();
    return uncheckedDetailedExitCode != null
        ? uncheckedDetailedExitCode
        : defaultDetailedExitCode();
  }

  private DetailedExitCode defaultDetailedExitCode() {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(Strings.nullToEmpty(getMessage()))
            .setPackageLoading(
                PackageLoading.newBuilder().setCode(PackageLoading.Code.TARGET_MISSING).build())
            .build());
  }
}
