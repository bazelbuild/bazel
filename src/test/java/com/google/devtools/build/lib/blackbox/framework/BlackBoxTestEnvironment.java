// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.blackbox.framework;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.MoreExecutors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Abstract class for setting up the blackbox test environment, returns {@link BlackBoxTestContext}
 * for the tests to access the environment and run Bazel/Blaze commands. This is the single entry
 * point in the blackbox tests framework for working with test environment.
 *
 * <p>The single instance of this class can be used for several tests. Each test should call {@link
 * #prepareEnvironment(String, ImmutableList)} to prepare environment and obtain the test context.
 *
 * <p>See {@link com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest}
 */
public abstract class BlackBoxTestEnvironment {
  private final ExecutorService executorService;
  // to control that the executor service is not used after it is shut down
  private boolean isDisposed;

  protected BlackBoxTestEnvironment() {
    executorService =
        MoreExecutors.getExitingExecutorService(
            (ThreadPoolExecutor) Executors.newFixedThreadPool(2), 1, TimeUnit.SECONDS);
  }

  protected abstract BlackBoxTestContext prepareEnvironment(
      String testName, ImmutableList<ToolsSetup> tools, ExecutorService executorService)
      throws Exception;

  /**
   * Prepares test environment and returns test context instance to be used by tests to access the
   * environment and invoke Bazel/Blaze commands.
   *
   * @param testName name of the current test, used to name the test working directory
   * @param tools the list of all tools setup classes {@link ToolsSetup} that should be called for
   *     the test
   * @return {@link BlackBoxTestContext} test context
   * @throws Exception if tools setup fails
   */
  public BlackBoxTestContext prepareEnvironment(String testName, ImmutableList<ToolsSetup> tools)
      throws Exception {
    Preconditions.checkState(!isDisposed);
    return prepareEnvironment(testName, tools, executorService);
  }

  /**
   * This method must be called when the test group execution is finished, for example, from
   * &#64;AfterClass method
   */
  public final void dispose() {
    MoreExecutors.shutdownAndAwaitTermination(executorService, 1, TimeUnit.SECONDS);
    isDisposed = true;
  }
}
