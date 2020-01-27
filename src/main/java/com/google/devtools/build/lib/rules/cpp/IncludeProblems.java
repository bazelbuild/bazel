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
import com.google.devtools.build.lib.actions.Artifact;

/**
 * Accumulator for problems encountered while reading or validating inclusion
 * results.
 */
class IncludeProblems {

  private StringBuilder message;  // null when no problems

  void add(String included) {
    if (message == null) { message = new StringBuilder(); }
    message.append("\n  '" + included + "'");
  }

  boolean hasProblems() { return message != null; }

  String getMessage(Action action, Artifact sourceFile) {
    if (message != null) {
      return "undeclared inclusion(s) in rule '" + action.getOwner().getLabel() + "':\n"
          + "this rule is missing dependency declarations for the following files "
          + "included by '" + sourceFile.prettyPrint() + "':"
          + message;
    }
    return null;
  }

  void assertProblemFree(Action action, Artifact sourceFile) throws ActionExecutionException {
    if (hasProblems()) {
      throw new ActionExecutionException(getMessage(action, sourceFile), action, false);
    }
  }
}
