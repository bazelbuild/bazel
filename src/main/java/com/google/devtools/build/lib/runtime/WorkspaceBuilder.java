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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutorFactory;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
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
  private Predicate<PathFragment> allowedMissingInputs;
  // We use an immutable map builder for the nice side effect that it throws if a duplicate key
  // is inserted.
  private final ImmutableMap.Builder<SkyFunctionName, SkyFunction> skyFunctions =
      ImmutableMap.builder();
  private final ImmutableList.Builder<PrecomputedValue.Injected> precomputedValues =
      ImmutableList.builder();
  private final ImmutableList.Builder<SkyValueDirtinessChecker> customDirtinessCheckers =
      ImmutableList.builder();

  WorkspaceBuilder(BlazeDirectories directories, BinTools binTools) {
    this.directories = directories;
    this.binTools = binTools;
  }

  BlazeWorkspace build(
      BlazeRuntime runtime,
      PackageFactory packageFactory,
      ConfiguredRuleClassProvider ruleClassProvider,
      String productName,
      SubscriberExceptionHandler eventBusExceptionHandler) throws AbruptExitException {
    // Set default values if none are set.
    if (skyframeExecutorFactory == null) {
      skyframeExecutorFactory = new SequencedSkyframeExecutorFactory();
    }
    if (allowedMissingInputs == null) {
      allowedMissingInputs = Predicates.alwaysFalse();
    }

    SkyframeExecutor skyframeExecutor = skyframeExecutorFactory.create(
        packageFactory,
        directories,
        binTools,
        workspaceStatusActionFactory,
        ruleClassProvider.getBuildInfoFactories(),
        diffAwarenessFactories.build(),
        allowedMissingInputs,
        skyFunctions.build(),
        precomputedValues.build(),
        customDirtinessCheckers.build(),
        productName);
    return new BlazeWorkspace(
        runtime, directories, skyframeExecutor, eventBusExceptionHandler,
        workspaceStatusActionFactory, binTools);
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

  /**
   * Action inputs are allowed to be missing for all inputs where this predicate returns true. Only
   * one predicate may be set per workspace.
   */
  public WorkspaceBuilder setAllowedMissingInputs(Predicate<PathFragment> allowedMissingInputs) {
    Preconditions.checkArgument(this.allowedMissingInputs == null,
        "At most one module may set allowed missing inputs. But found two: %s and %s",
        this.allowedMissingInputs, allowedMissingInputs);
    this.allowedMissingInputs = Preconditions.checkNotNull(allowedMissingInputs);
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

  /**
   * Adds an extra precomputed value to Skyframe.
   *
   * <p>This functionality can be used to implement precomputed values that are not constant during
   * the lifetime of a Blaze instance (naturally, they must be constant over the course of a build).
   *
   * <p>The following things must be done in order to define a new precomputed values:
   * <ul>
   * <li> Create a public static final variable of type
   *     {@link com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed}.
   * <li> Set its value by adding an {@link Injected} via this method (it can be created using the
   *     aforementioned variable and the value or a supplier of the value).
   * <li> Reference the value in Skyframe functions by calling the {@code get} method on the
   *     {@link com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed} variable.
   * </ul>
   */
  public WorkspaceBuilder addPrecomputedValue(PrecomputedValue.Injected precomputedValue) {
    this.precomputedValues.add(Preconditions.checkNotNull(precomputedValue));
    return this;
  }

  public WorkspaceBuilder addCustomDirtinessChecker(
      SkyValueDirtinessChecker customDirtinessChecker) {
    this.customDirtinessCheckers.add(Preconditions.checkNotNull(customDirtinessChecker));
    return this;
  }
}
