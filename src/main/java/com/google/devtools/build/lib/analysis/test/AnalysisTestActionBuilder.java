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
import com.google.devtools.build.lib.util.OS;

/**
 * Helper for writing test actions for analysis test rules. Analysis test rules are
 * restricted to disallow the rule implementation functions from registering actions themselves;
 * such rules register test success/failure via {@link AnalysisTestResultInfo}. This helper
 * registers the appropriate test script simulating success or failure of the test.
 */
public class AnalysisTestActionBuilder {

  /**
   * Register and return an action to write a test script to the default executable location
   * reflecting the given info object.
   */
  public static FileWriteAction writeAnalysisTestAction(
      RuleContext ruleContext,
      AnalysisTestResultInfo infoObject) {
    FileWriteAction action;
    // TODO(laszlocsomor): Use the execution platform, not the host platform.
    boolean isExecutedOnWindows = OS.getCurrent() == OS.WINDOWS;

    if (isExecutedOnWindows) {
      StringBuilder sb = new StringBuilder().append("@echo off\n");
      for (String line : Splitter.on("\n").split(infoObject.getMessage())) {
        sb.append("echo ").append(line).append("\n");
      }
      String content = sb
          .append("exit /b ").append(infoObject.getSuccess() ? "0" : "1")
          .toString();

      action = FileWriteAction.create(ruleContext,
          ruleContext.createOutputArtifactScript(), content, /* executable */ true);

    } else {
      String content =
          "cat << EOF\n"
              + infoObject.getMessage()
              + "\n"
              + "EOF\n"
              + "exit "
              + (infoObject.getSuccess() ? "0" : "1");
      action = FileWriteAction.create(ruleContext,
          ruleContext.createOutputArtifactScript(), content, /* executable */ true);
    }

    ruleContext.registerAction(action);
    return action;
  }
}
