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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.concurrent.Immutable;

/** Base implementation of a Spawn. */
@Immutable
public class BaseSpawn implements Spawn {
  private final ImmutableList<String> arguments;
  private final ImmutableMap<String, String> environment;
  private final ImmutableMap<String, String> executionInfo;
  private final ImmutableSet<PathFragment> optionalOutputFiles;
  private final RunfilesSupplier runfilesSupplier;
  private final ActionExecutionMetadata action;
  private final ResourceSet localResources;

  public BaseSpawn(
      List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier,
      ActionExecutionMetadata action,
      ResourceSet localResources,
      Collection<PathFragment> optionalOutputFiles) {
    this.arguments = ImmutableList.copyOf(arguments);
    this.environment = ImmutableMap.copyOf(environment);
    this.executionInfo = ImmutableMap.copyOf(executionInfo);
    this.runfilesSupplier = runfilesSupplier;
    this.action = action;
    this.localResources = localResources;
    this.optionalOutputFiles = ImmutableSet.copyOf(optionalOutputFiles);
  }

  /**
   * Returns a new Spawn. The caller must not modify the parameters after the call; neither will
   * this method.
   */
  public BaseSpawn(
      List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier,
      ActionExecutionMetadata action,
      ResourceSet localResources) {
    this(
        arguments,
        environment,
        executionInfo,
        runfilesSupplier,
        action,
        localResources,
        ImmutableSet.<PathFragment>of());
  }

  /** Returns a new Spawn. */
  public BaseSpawn(
      List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      ActionExecutionMetadata action,
      ResourceSet localResources) {
    this(
        arguments,
        environment,
        executionInfo,
        EmptyRunfilesSupplier.INSTANCE,
        action,
        localResources);
  }

  public static PathFragment runfilesForFragment(PathFragment pathFragment) {
    return pathFragment.getParentDirectory().getChild(pathFragment.getBaseName() + ".runfiles");
  }

  @Override
  public boolean hasNoSandbox() {
    return executionInfo.containsKey("nosandbox");
  }

  @Override
  public boolean isRemotable() {
    return !executionInfo.containsKey("local");
  }

  @Override
  public final ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public RunfilesSupplier getRunfilesSupplier() {
    return runfilesSupplier;
  }

  @Override
  public ImmutableList<Artifact> getFilesetManifests() {
    return ImmutableList.<Artifact>of();
  }

  @Override
  public ImmutableList<String> getArguments() {
    // TODO(bazel-team): this method should be final, as the correct value of the args can be
    // injected in the ctor.
    return arguments;
  }

  @Override
  public ImmutableMap<String, String> getEnvironment() {
    PathFragment runfilesRoot = getRunfilesRoot();
    if (runfilesRoot == null
        || (environment.containsKey("JAVA_RUNFILES")
            && environment.containsKey("PYTHON_RUNFILES"))) {
      return environment;
    } else {
      ImmutableMap.Builder<String, String> env = ImmutableMap.builder();
      env.putAll(environment);
      // TODO(bazel-team): Unify these into a single env variable.
      String runfilesRootString = runfilesRoot.getPathString();
      env.put("JAVA_RUNFILES", runfilesRootString);
      env.put("PYTHON_RUNFILES", runfilesRootString);
      return env.build();
    }
  }

  /** @return the runfiles directory if there is only one, otherwise null */
  private PathFragment getRunfilesRoot() {
    Set<PathFragment> runfilesSupplierRoots = runfilesSupplier.getRunfilesDirs();
    if (runfilesSupplierRoots.size() == 1) {
      return Iterables.getOnlyElement(runfilesSupplierRoots);
    } else {
      return null;
    }
  }

  @Override
  public Iterable<? extends ActionInput> getToolFiles() {
    return action.getTools();
  }

  @Override
  public Iterable<? extends ActionInput> getInputFiles() {
    return action.getInputs();
  }

  @Override
  public Collection<? extends ActionInput> getOutputFiles() {
    return action.getOutputs();
  }

  @Override
  public Collection<PathFragment> getOptionalOutputFiles() {
    return optionalOutputFiles;
  }

  @Override
  public ActionExecutionMetadata getResourceOwner() {
    return action;
  }

  @Override
  public ResourceSet getLocalResources() {
    return localResources;
  }

  @Override
  public String getMnemonic() {
    return action.getMnemonic();
  }

  /** A local spawn. */
  public static class Local extends BaseSpawn {
    public Local(
        List<String> arguments, Map<String, String> environment, ActionExecutionMetadata action,
        ResourceSet localResources) {
      this(arguments, environment, ImmutableMap.<String, String>of(), action, localResources);
    }

    public Local(
        List<String> arguments,
        Map<String, String> environment,
        Map<String, String> executionInfo,
        ActionExecutionMetadata action,
        ResourceSet localResources) {
      super(arguments, environment, buildExecutionInfo(executionInfo), action, localResources);
    }

    private static ImmutableMap<String, String> buildExecutionInfo(
        Map<String, String> additionalExecutionInfo) {
      ImmutableMap.Builder<String, String> executionInfo = ImmutableMap.builder();
      executionInfo.putAll(additionalExecutionInfo);
      executionInfo.put("local", "");
      return executionInfo.build();
    }
  }
}
