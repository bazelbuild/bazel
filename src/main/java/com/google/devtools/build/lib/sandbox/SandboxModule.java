// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/**
 * This module provides the Sandbox spawn strategy.
 */
public final class SandboxModule extends BlazeModule {
  private Path sandboxBase;
  private boolean shouldCleanupSandboxBase;

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.<Class<? extends OptionsBase>>of(SandboxOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void executorInit(
      CommandEnvironment cmdEnv, BuildRequest request, ExecutorBuilder builder) {
    BlazeDirectories blazeDirs = cmdEnv.getDirectories();
    String productName = cmdEnv.getRuntime().getProductName();
    SandboxOptions sandboxOptions = request.getOptions(SandboxOptions.class);
    FileSystem fs = blazeDirs.getFileSystem();

    if (sandboxOptions.sandboxBase.isEmpty()) {
      sandboxBase = blazeDirs.getOutputBase().getRelative(productName + "-sandbox");
    } else {
      String dirName =
          productName + "-sandbox." + Fingerprint.md5Digest(blazeDirs.getOutputBase().toString());
      sandboxBase = fs.getPath(sandboxOptions.sandboxBase).getRelative(dirName);
    }

    // Do not remove the sandbox base when --sandbox_debug was specified so that people can check
    // out the contents of the generated sandbox directories.
    shouldCleanupSandboxBase = !sandboxOptions.sandboxDebug;

    try {
      FileSystemUtils.createDirectoryAndParents(sandboxBase);
      builder.addActionContextProvider(
          SandboxActionContextProvider.create(cmdEnv, request, sandboxBase));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    builder.addActionContextConsumer(new SandboxActionContextConsumer(cmdEnv));
  }

  @Override
  public void afterCommand() {
    super.afterCommand();

    if (sandboxBase != null) {
      if (shouldCleanupSandboxBase) {
        try {
          FileSystemUtils.deleteTree(sandboxBase);
        } catch (IOException e) {
          // Nothing we can do at this point.
        }
      }
      sandboxBase = null;
    }
  }
}
