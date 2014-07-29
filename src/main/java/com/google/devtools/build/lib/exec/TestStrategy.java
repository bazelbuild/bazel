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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.test.TestActionContext;
import com.google.devtools.build.lib.view.test.TestRunnerAction;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.HashMap;
import java.util.Map;

/**
 * A strategy for executing a {@link TestRunnerAction}.
 */
public abstract class TestStrategy implements TestActionContext {
  /**
   * Converter for the --flaky_test_attempts option.
   */
  public static class TestAttemptsConverter extends RangeConverter {
    public TestAttemptsConverter() {
      super(1, 10);
    }

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      if ("default".equals(input)) {
        return -1;
      } else {
        return super.convert(input);
      }
    }

    @Override
    public String getTypeDescription() {
      return super.getTypeDescription() + " or the string \"default\"";
    }
  }

  public enum TestOutputFormat {
    SUMMARY, // Provide summary output only.
    ERRORS, // Print output from failed tests to the stderr after the test failure.
    ALL, // Print output from all tests to the stderr after the test completion.
    STREAMED; // Stream output for each test.

    /**
     * Converts to {@link TestOutputFormat}.
     */
    public static class Converter extends EnumConverter<TestOutputFormat> {
      public Converter() {
        super(TestOutputFormat.class, "test output");
      }
    }
  }

  public enum TestSummaryFormat {
    SHORT, // Print information only about tests.
    TERSE, // Like "SHORT", but even shorter: Do not print PASSED tests.
    DETAILED, // Print information only about failed test cases.
    NONE; // Do not print summary.

    /**
     * Converts to {@link TestSummaryFormat}.
     */
    public static class Converter extends EnumConverter<TestSummaryFormat> {
      public Converter() {
        super(TestSummaryFormat.class, "test summary");
      }
    }
  }

  public static final PathFragment TEST_TMP_ROOT = new PathFragment("_tmp");

  // Used for selecting subset of testcase / testmethods.
  private static final String TEST_BRIDGE_TEST_FILTER_ENV = "TESTBRIDGE_TEST_ONLY";

  protected final ExecutionOptions executionOptions;

  protected final OutErr outErr;

  public TestStrategy(OptionsClassProvider options, OutErr outErr) {
    this.executionOptions = options.getOptions(ExecutionOptions.class);
    this.outErr = outErr;
  }

  /** The strategy name, preferably suitable for passing to --test_strategy. */
  public abstract String testStrategyName();

  @Override
  public abstract void exec(TestRunnerAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException;

  @Override
  public abstract String strategyLocality(TestRunnerAction action);

  /**
   * Callback for determining the strategy locality.
   *
   * @param action the test action
   * @param localRun whether to run it locally
   */
  protected String strategyLocality(TestRunnerAction action, boolean localRun) {
    return strategyLocality(action);
  }

  /**
   * Returns mutable map of default testing shell environment. By itself it is incomplete and is
   * modified further by the specific test strategy implementations (mostly due to the fact that
   * environments used locally and remotely are different).
   */
  protected Map<String, String> getDefaultTestEnvironment(TestRunnerAction action) {
    Map<String, String> env = new HashMap<>();

    env.putAll(action.getConfiguration().getDefaultShellEnvironment());
    env.remove("LANG");
    env.put("TZ", "UTC");
    env.put("TEST_SIZE", action.getTestProperties().getSize().toString());
    env.put("TEST_TIMEOUT", Integer.toString(getTimeout(action)));

    if (action.isSharded()) {
      env.put("TEST_SHARD_INDEX", Integer.toString(action.getShardNum()));
      env.put("TEST_TOTAL_SHARDS",
          Integer.toString(action.getExecutionSettings().getTotalShards()));
    }

    // When we run test multiple times, set different TEST_RANDOM_SEED values for each run.
    if (action.getConfiguration().getRunsPerTestForLabel(action.getOwner().getLabel()) > 1) {
      env.put("TEST_RANDOM_SEED", Integer.toString(action.getRunNumber() + 1));
    }

    String testFilter = action.getExecutionSettings().getTestFilter();
    if (testFilter != null) {
      env.put(TEST_BRIDGE_TEST_FILTER_ENV, testFilter);
    }

    return env;
  }

  /**
   * Returns the number of attempts specific test action can be retried.
   *
   * <p>For rules with "flaky = 1" attribute, this method will return 3 unless --flaky_test_attempts
   * option is given and specifies another value.
   */
  @VisibleForTesting /* protected */
  public int getTestAttempts(TestRunnerAction action) {
    if (executionOptions.testAttempts == -1) {
      return action.getTestProperties().isFlaky() ? 3 : 1;
    } else {
      return executionOptions.testAttempts;
    }
  }

  /**
   * Returns timeout value in seconds that should be used for the given test action. We always use
   * the "categorical timeouts" which are based on the --test_timeout flag. A rule picks its timeout
   * but ends up with the same effective value as all other rules in that bucket.
   */
  protected final int getTimeout(TestRunnerAction testAction) {
    return executionOptions.testTimeout.get(testAction.getTestProperties().getTimeout());
  }

  /**
   * Returns a subset of the environment from the current shell.
   *
   * <p>Warning: Since these variables are not part of the configuration's fingerprint, they
   * MUST NOT be used by any rule or action in such a way as to affect the semantics of that
   * build step.
   */
  public Map<String, String> getAdmissibleShellEnvironment(BuildConfiguration config,
      Iterable<String> variables) {
    return getMapping(variables, config.getClientEnv());
  }

  /**
   * For an given environment, returns a subset containing all variables in the given list if they
   * are defined in the given environment.
   */
  @VisibleForTesting
  static Map<String, String> getMapping(Iterable<String> variables,
      Map<String, String> environment) {
    Map<String, String> result = new HashMap<>();
    for (String var : variables) {
      if (environment.containsKey(var)) {
        result.put(var, environment.get(var));
      }
    }
    return result;
  }
}
