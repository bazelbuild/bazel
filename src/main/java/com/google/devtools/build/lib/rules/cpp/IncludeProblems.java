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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.server.FailureDetails.CppCompile;
import com.google.devtools.build.lib.server.FailureDetails.CppCompile.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;

/**
 * Accumulator for problems encountered while reading or validating inclusion
 * results.
 */
class IncludeProblems {

  private StringBuilder problems; // null when no problems

  void add(String included) {
    if (problems == null) {
      problems = new StringBuilder();
    }
    problems.append("\n  '").append(included).append("'");
  }

  boolean hasProblems() {
    return problems != null;
  }

  void assertProblemFree(String message, Action action) throws ActionExecutionException {
    if (hasProblems()) {
      String fullMessage = message + problems;
      DetailedExitCode code =
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(fullMessage)
                  .setCppCompile(CppCompile.newBuilder().setCode(Code.UNDECLARED_INCLUSIONS))
                  .build());
      throw new ActionExecutionException(fullMessage, action, false, code);
    }
  }
}
