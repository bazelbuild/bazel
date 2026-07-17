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
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/** Exception indicating an attempt to access something which is not found or does not exist. */
public class NoSuchThingException extends Exception implements DetailedException {

  // TODO(b/138456686): Remove Nullable and add Precondition#checkNotNull in constructor when all
  //  subclasses are instantiated with DetailedExitCode.
  @Nullable private final DetailedExitCode detailedExitCode;

  public NoSuchThingException(String message) {
    super(message);
    this.detailedExitCode = null;
  }

  NoSuchThingException(String message, Throwable cause) {
    super(message, cause);
    this.detailedExitCode = null;
  }

  NoSuchThingException(String message, DetailedExitCode detailedExitCode) {
    super(message);
    this.detailedExitCode = detailedExitCode;
  }

  NoSuchThingException(String message, Throwable cause, DetailedExitCode detailedExitCode) {
    super(message, cause);
    this.detailedExitCode = detailedExitCode;
  }

  /**
   * Returns the detail exit code if it exists. If it does not exist, then return the default
   * detailed exit code.
   */
  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode != null ? detailedExitCode : defaultDetailedExitCode();
  }

  /** Returns the detailed exit code but does not check if it is null. */
  @Nullable
  DetailedExitCode getUncheckedDetailedExitCode() {
    return detailedExitCode;
  }

  private DetailedExitCode defaultDetailedExitCode() {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(Strings.nullToEmpty(getMessage()))
            .setPackageLoading(
                PackageLoading.newBuilder().setCode(PackageLoading.Code.NO_SUCH_THING).build())
            .build());
  }
}
