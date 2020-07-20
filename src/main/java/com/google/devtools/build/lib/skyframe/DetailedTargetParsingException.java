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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.util.DetailedExitCode;

/** A {@link TargetParsingException} with {@link DetailedExitCode}. */
public class DetailedTargetParsingException extends TargetParsingException
    implements DetailedException {

  private final DetailedExitCode detailedExitCode;

  public DetailedTargetParsingException(
      Throwable cause, String message, DetailedExitCode detailedExitCode) {
    super(message, cause);
    this.detailedExitCode = detailedExitCode;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
