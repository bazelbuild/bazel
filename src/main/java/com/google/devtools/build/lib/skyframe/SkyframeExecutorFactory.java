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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.Factory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;

/**
* A factory that creates instances of SkyframeExecutor.
*/
public interface SkyframeExecutorFactory {

  /**
   * Creates an instance of SkyframeExecutor
   *
   * @param tsgm timestamp granularity monitor
   * @param pkgFactory the package factory
   * @param directories Blaze directories
   * @param binTools the embedded tools
   * @param workspaceStatusActionFactory a factory for creating WorkspaceStatusAction objects
   * @param buildInfoFactories list of BuildInfoFactories
   * @param diffAwarenessFactories
   * @param allowedMissingInputs
   * @param extraSkyFunctions
   * @param extraPrecomputedValues
   * @param customDirtinessCheckers
   * @return an instance of the SkyframeExecutor
   * @throws AbruptExitException if the executor cannot be created
   */
  SkyframeExecutor create(
      PackageFactory pkgFactory,
      BlazeDirectories directories,
      BinTools binTools,
      Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions,
      ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues,
      Iterable<SkyValueDirtinessChecker> customDirtinessCheckers)
      throws AbruptExitException;
}
