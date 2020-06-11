// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.profiler.memory.AllocationTracker;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.ManagedDirectoriesKnowledge;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutorFactory;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.Map;

/**
 * Builder class to create a {@link BlazeWorkspace} instance. This class is part of the module API,
 * which allows modules to affect how the workspace is initialized.
 */
public final class WorkspaceBuilder {
  private final BlazeDirectories directories;
  private final BinTools binTools;

  private SkyframeExecutorFactory skyframeExecutorFactory;
  private WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  private final ImmutableList.Builder<DiffAwareness.Factory> diffAwarenessFactories =
      ImmutableList.builder();
  // We use an immutable map builder for the nice side effect that it throws if a duplicate key
  // is inserted.
  private final ImmutableMap.Builder<SkyFunctionName, SkyFunction> skyFunctions =
      ImmutableMap.builder();
  private final ImmutableList.Builder<SkyValueDirtinessChecker> customDirtinessCheckers =
      ImmutableList.builder();
  private AllocationTracker allocationTracker;
  private ManagedDirectoriesKnowledge managedDirectoriesKnowledge;

  WorkspaceBuilder(BlazeDirectories directories, BinTools binTools) {
    this.directories = directories;
    this.binTools = binTools;
  }

  BlazeWorkspace build(
      BlazeRuntime runtime,
      PackageFactory packageFactory,
      ConfiguredRuleClassProvider ruleClassProvider,
      SubscriberExceptionHandler eventBusExceptionHandler) throws AbruptExitException {
    // Set default values if none are set.
    if (skyframeExecutorFactory == null) {
      skyframeExecutorFactory =
          new SequencedSkyframeExecutorFactory(runtime.getDefaultBuildOptions());
    }

    SkyframeExecutor skyframeExecutor =
        skyframeExecutorFactory.create(
            packageFactory,
            runtime.getFileSystem(),
            directories,
            runtime.getActionKeyContext(),
            workspaceStatusActionFactory,
            diffAwarenessFactories.build(),
            skyFunctions.build(),
            customDirtinessCheckers.build(),
            managedDirectoriesKnowledge);
    return new BlazeWorkspace(
        runtime,
        directories,
        skyframeExecutor,
        eventBusExceptionHandler,
        workspaceStatusActionFactory,
        binTools,
        allocationTracker);
  }

  /**
   * Sets a factory for creating {@link SkyframeExecutor} objects. Note that only one factory per
   * workspace is allowed.
   */
  public WorkspaceBuilder setSkyframeExecutorFactory(
      SkyframeExecutorFactory skyframeExecutorFactory) {
    Preconditions.checkState(this.skyframeExecutorFactory == null,
        "At most one Skyframe factory supported. But found two: %s and %s",
        this.skyframeExecutorFactory, skyframeExecutorFactory);
    this.skyframeExecutorFactory = Preconditions.checkNotNull(skyframeExecutorFactory);
    return this;
  }

  /**
   * Sets the workspace status action factory contributed by this module. Only one factory per
   * workspace is allowed.
   */
  public WorkspaceBuilder setWorkspaceStatusActionFactory(
      WorkspaceStatusAction.Factory workspaceStatusActionFactory) {
    Preconditions.checkState(this.workspaceStatusActionFactory == null,
        "At most one workspace status action factory supported. But found two: %s and %s",
        this.workspaceStatusActionFactory, workspaceStatusActionFactory);
    this.workspaceStatusActionFactory = Preconditions.checkNotNull(workspaceStatusActionFactory);
    return this;
  }

  public WorkspaceBuilder setAllocationTracker(AllocationTracker allocationTracker) {
    Preconditions.checkState(
        this.allocationTracker == null, "At most one allocation tracker can be set.");
    this.allocationTracker = Preconditions.checkNotNull(allocationTracker);
    return this;
  }

  /**
   * Add a {@link DiffAwareness} factory. These will be used to determine which files, if any,
   * changed between Blaze commands. Note that these factories are attempted in the order in which
   * they are added to this class, so order matters - in order to guarantee a specific order, only
   * a single module should add such factories.
   */
  public WorkspaceBuilder addDiffAwarenessFactory(DiffAwareness.Factory factory) {
    this.diffAwarenessFactories.add(Preconditions.checkNotNull(factory));
    return this;
  }

  /** Add an "extra" SkyFunction for SkyValues. */
  public WorkspaceBuilder addSkyFunction(SkyFunctionName name, SkyFunction skyFunction) {
    Preconditions.checkNotNull(name);
    Preconditions.checkNotNull(skyFunction);
    this.skyFunctions.put(name, skyFunction);
    return this;
  }

  /** Add "extra" SkyFunctions for SkyValues. */
  public WorkspaceBuilder addSkyFunctions(Map<SkyFunctionName, SkyFunction> skyFunctions) {
    this.skyFunctions.putAll(Preconditions.checkNotNull(skyFunctions));
    return this;
  }

  public WorkspaceBuilder addCustomDirtinessChecker(
      SkyValueDirtinessChecker customDirtinessChecker) {
    this.customDirtinessCheckers.add(Preconditions.checkNotNull(customDirtinessChecker));
    return this;
  }

  public WorkspaceBuilder setManagedDirectoriesKnowledge(
      ManagedDirectoriesKnowledge managedDirectoriesKnowledge) {
    this.managedDirectoriesKnowledge = managedDirectoriesKnowledge;
    return this;
  }
}
