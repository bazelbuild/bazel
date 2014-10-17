// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.WorkspaceStatusAction;
import com.google.devtools.build.lib.view.buildinfo.BuildInfoFactory;

/**
* A factory that creates instances of SkyframeExecutor.
*/
public interface SkyframeExecutorFactory {

  /**
   * Creates an instance of SkyframeExecutor
   *
   * @param reporter the reporter to be used by the executor
   * @param pkgFactory the package factory
   * @param skyframeBuild use Skyframe for the build phase. Should be always true after we are in
   * the skyframe full mode.
   * @param tsgm timestamp granularity monitor
   * @param directories Blaze directories
   * @param workspaceStatusActionFactory a factory for creating WorkspaceStatusAction objects
   * @param buildInfoFactories list of BuildInfoFactories
   * @param diffAwarenessFactories
   * @param allowedMissingInputs
   * @param preprocessorFactorySupplier
   * @param clock
   * @return an instance of the SkyframeExecutor
   */
  SkyframeExecutor create(Reporter reporter, PackageFactory pkgFactory,
      boolean skyframeBuild, TimestampGranularityMonitor tsgm,
      BlazeDirectories directories,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      ImmutableList<BuildInfoFactory> buildInfoFactories,
      Iterable<? extends DiffAwareness.Factory> diffAwarenessFactories,
      Predicate<PathFragment> allowedMissingInputs,
      Preprocessor.Factory.Supplier preprocessorFactorySupplier,
      Clock clock);
}
