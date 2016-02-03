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
package com.google.devtools.build.lib.rules.test;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;

import java.io.IOException;

/**
 * A context for the execution of test actions ({@link TestRunnerAction}).
 */
public interface TestActionContext extends ActionContext {

  /**
   * Executes the test command, directing standard out / err to {@code outErr}.  The status of
   * the test should be communicated by posting a {@link TestResult} object to the eventbus.
   */
  void exec(TestRunnerAction action,
      ActionExecutionContext actionExecutionContext) throws ExecException, InterruptedException;

  /**
   * Creates a cached test result.
   */
  TestResult newCachedTestResult(Path execRoot, TestRunnerAction action, TestResultData cached)
      throws IOException;
}
