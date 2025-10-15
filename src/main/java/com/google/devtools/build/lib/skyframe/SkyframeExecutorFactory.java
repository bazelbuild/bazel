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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.function.Supplier;

/** A factory that creates instances of SkyframeExecutor. */
public interface SkyframeExecutorFactory {

  /**
   * Creates an instance of SkyframeExecutor
   *
   * @param pkgFactory the package factory
   * @param fileSystem the Blaze file system
   * @param directories Blaze directories
   * @param workspaceStatusActionFactory a factory for creating WorkspaceStatusAction objects
   * @param skyKeyStateReceiver a receiver for SkyKeys as they start evaluating and are evaluated
   * @param bugReporter BugReporter: always BugReporter.defaultInstance() outside of Java tests
   * @return an instance of the SkyframeExecutor
   * @throws AbruptExitException if the executor cannot be created
   */
  SkyframeExecutor create(
      PackageFactory pkgFactory,
      FileSystem fileSystem,
      BlazeDirectories directories,
      ActionKeyContext actionKeyContext,
      Factory workspaceStatusActionFactory,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      SyscallCache syscallCache,
      boolean allowExternalRepositories,
      Supplier<Path> repoContentsCachePathSupplier,
      SkyframeExecutor.SkyKeyStateReceiver skyKeyStateReceiver,
      BugReporter bugReporter)
      throws AbruptExitException;
}
