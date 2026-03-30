// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import static java.lang.Math.min;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

/** Utility for formatting error messages. */
public final class ErrorMessageHelper {
  @VisibleForTesting static final int MAX_ERRORS_TO_REPORT = 5;

  public static String getErrorMessage(ImmutableList<Throwable> errors) {
    var message = new StringBuilder();
    if (errors.size() > 1) {
      message.append("There were ").append(errors.size()).append(" write errors.");
      if (errors.size() > MAX_ERRORS_TO_REPORT) {
        message
            .append(" Only the first ")
            .append(MAX_ERRORS_TO_REPORT)
            .append(" will be reported.");
      }
      message.append('\n');
    }
    for (int i = 0; i < min(errors.size(), MAX_ERRORS_TO_REPORT); i++) {
      message.append(errors.get(i).getMessage()).append('\n');
    }
    return message.toString();
  }

  private ErrorMessageHelper() {}
}
