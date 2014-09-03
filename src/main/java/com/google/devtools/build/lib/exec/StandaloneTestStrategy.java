// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.view.config.BinTools;
import com.google.devtools.build.lib.view.test.TestActionContext;
import com.google.devtools.build.lib.view.test.TestResult;
import com.google.devtools.build.lib.view.test.TestRunnerAction;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.OptionsClassProvider;

/**
 * Runs TestRunnerAction actions. 
 */
@ExecutionStrategy(contextType = TestActionContext.class,
          name = { "standalone" })
public class StandaloneTestStrategy extends TestStrategy {
  public StandaloneTestStrategy(OptionsClassProvider options, BinTools binTools) {
    super(options, binTools);
  }

  public String testStrategyName() { return "standalone"; }

  @Override
  public void exec(TestRunnerAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    throw new TestExecException("exec not yet implemented.");
  }

  @Override
  public String strategyLocality(TestRunnerAction action) { return "standalone"; }

  @Override
  public TestResult newCachedTestResult(TestRunnerAction action, TestResultData data) {
    return new TestResult(action, data, /*cached*/ true);
  }
}
