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
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import java.io.IOException;

/**
 * Test strategy wrapper called 'exclusive'. It should delegate to a test strategy for local
 * execution. The name 'exclusive' triggers behavior it triggers behavior in
 * SkyframeExecutor to schedule test execution sequentially after non-test actions. This
 * ensures streamed test output is not polluted by other action output.
 */
@ExecutionStrategy(contextType = TestActionContext.class,
          name = { "exclusive" })
public class ExclusiveTestStrategy implements TestActionContext {
  private TestActionContext parent;

  public ExclusiveTestStrategy(TestActionContext parent) {
    this.parent = parent;
  }

  @Override
  public TestRunnerSpawn createTestRunnerSpawn(
      TestRunnerAction testRunnerAction, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    return parent.createTestRunnerSpawn(testRunnerAction, actionExecutionContext);
  }

  @Override
  public boolean isTestKeepGoing() {
    return parent.isTestKeepGoing();
  }

  @Override
  public TestResult newCachedTestResult(
      Path execRoot, TestRunnerAction action, TestResultData cached) throws IOException {
    return parent.newCachedTestResult(execRoot, action, cached);
  }
}
