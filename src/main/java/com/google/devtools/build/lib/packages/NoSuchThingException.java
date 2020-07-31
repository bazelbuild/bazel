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

  public NoSuchThingException(String message, Throwable cause) {
    super(message, cause);
    this.detailedExitCode = null;
  }

  public NoSuchThingException(String message, DetailedExitCode detailedExitCode) {
    super(message);
    this.detailedExitCode = detailedExitCode;
  }

  public NoSuchThingException(String message, Throwable cause, DetailedExitCode detailedExitCode) {
    super(message, cause);
    this.detailedExitCode = detailedExitCode;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
