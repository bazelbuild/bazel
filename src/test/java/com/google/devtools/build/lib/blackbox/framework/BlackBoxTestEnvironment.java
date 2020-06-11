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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.MoreExecutors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

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
  /**
   * Executor service for reading stdout and stderr streams of the process. Has exactly two threads
   * since there are two streams.
   */
  @Nullable
  private ExecutorService executorService =
      MoreExecutors.getExitingExecutorService(
          (ThreadPoolExecutor) Executors.newFixedThreadPool(2),
          1, TimeUnit.SECONDS);

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
    Preconditions.checkNotNull(executorService);
    return prepareEnvironment(testName, tools, executorService);
  }

  /**
   * This method must be called when the test group execution is finished, for example, from
   * &#64;AfterClass method.
   */
  public final void dispose() {
    Preconditions.checkNotNull(executorService);
    MoreExecutors.shutdownAndAwaitTermination(executorService, 1, TimeUnit.SECONDS);
    executorService = null;
  }

  public static String getWorkspaceWithDefaultRepos() {
    return Joiner.on("\n")
        .join(
            "load('@bazel_tools//tools/build_defs/repo:http.bzl', 'http_archive')",
            "http_archive(",
            "    name = 'rules_cc',",
            "    sha256 = '1d4dbbd1e1e9b57d40bb0ade51c9e882da7658d5bfbf22bbd15b68e7879d761f',",
            "    strip_prefix = 'rules_cc-8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0',",
            "    urls = [",
            "        'https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/"
                + "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip',",
            "        'https://github.com/bazelbuild/rules_cc/archive/"
                + "8bd6cd75d03c01bb82561a96d9c1f9f7157b13d0.zip',",
            "    ],",
            ")",
            "http_archive(",
            "    name = 'rules_proto',",
            "    sha256 = '602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208',",
            "    strip_prefix = 'rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313',",
            "    urls = [",
            "        'https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/"
                + "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz',",
            "        'https://github.com/bazelbuild/rules_proto/archive/"
                + "97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz',",
            "    ],",
            ")");
  }
}
