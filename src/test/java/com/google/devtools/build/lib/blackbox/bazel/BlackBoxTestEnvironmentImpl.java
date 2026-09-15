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

package com.google.devtools.build.lib.blackbox.bazel;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestEnvironment;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ToolsSetup;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * Implementation of {@link BlackBoxTestEnvironment} with the code of initializing Bazel blackbox
 * test environment.
 */
public class BlackBoxTestEnvironmentImpl extends BlackBoxTestEnvironment {
  @Override
  public BlackBoxTestContext prepareEnvironment(
      String testName, ImmutableList<ToolsSetup> tools, ExecutorService executorService)
      throws Exception {
    Path binaryPath = RunfilesUtil.find("io_bazel/src/bazel");

    BlackBoxTestContext testContext =
        new BlackBoxTestContext(
            testName, "bazel", binaryPath, Collections.emptyMap(), executorService);
    // Any Bazel command requires that workspace is already set up.
    testContext.write("MODULE.bazel");
    Path defaultLockfile = RunfilesUtil.find("io_bazel/src/test/tools/bzlmod/MODULE.bazel.lock");
    Files.copy(defaultLockfile, testContext.getWorkDir().resolve("MODULE.bazel.lock"));

    List<ToolsSetup> allTools = Lists.newArrayList(new DefaultToolsSetup());
    allTools.addAll(tools);
    for (ToolsSetup tool : allTools) {
      tool.setup(testContext);
    }

    PathUtils.setTreeWritable(testContext.getWorkDir());

    return testContext;
  }
}
