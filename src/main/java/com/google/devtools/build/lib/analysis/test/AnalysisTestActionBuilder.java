
// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;

/**
 * Helper for writing test actions for analysis test rules. Analysis test rules are restricted to
 * disallow the rule implementation functions from registering actions themselves; such rules
 * register test success/failure via {@link AnalysisTestResultInfo}. This helper registers the
 * appropriate test script simulating success or failure of the test.
 */
public class AnalysisTestActionBuilder {

  private AnalysisTestActionBuilder() {}

  /**
   * Register an action to write a test script to the default executable location. The script should
   * return exit status 0 if the test passed. It should print the failure message and return exit
   * status 1 if the test failed.
   */
  public static void writeAnalysisTestAction(
      RuleContext ruleContext, AnalysisTestResultInfo testResultInfo) {
    String escapedMessage =
        ruleContext.isExecutedOnWindows()
            ? testResultInfo.getMessage().replace("%", "%%")
            // Prefix each character with \ (double-escaped; once in the string, once in the
            // replacement sequence, which allows backslash-escaping literal "$"). "." is put in
            // parentheses because ErrorProne is overly vigorous about objecting to "." as an
            // always-matching regex (b/201772278).
            : testResultInfo.getMessage().replaceAll("(.)", "\\\\$1");
    StringBuilder sb = new StringBuilder();
    if (ruleContext.isExecutedOnWindows()) {
      sb.append("@echo off\n");
    }
    for (String line : Splitter.on("\n").split(escapedMessage)) {
      sb.append("echo ").append(line).append("\n");
    }
    sb.append("exit ");
    if (ruleContext.isExecutedOnWindows()) {
      sb.append("/b ");
    }
    sb.append(testResultInfo.getSuccess() ? "0" : "1");
    FileWriteAction action =
        FileWriteAction.create(
            ruleContext,
            ruleContext.createOutputArtifactScript(),
            sb.toString(),
            /*makeExecutable=*/ true);
    ruleContext.registerAction(action);
  }
}
