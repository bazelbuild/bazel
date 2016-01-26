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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.extra.EnvironmentVariable;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.util.CommandDescriptionForm;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.concurrent.Immutable;

/**
 * Base implementation of a Spawn.
 */
@Immutable
public class BaseSpawn implements Spawn {
  private final ImmutableList<String> arguments;
  private final ImmutableMap<String, String> environment;
  private final ImmutableMap<String, String> executionInfo;
  private final ImmutableMap<PathFragment, Artifact> runfilesManifests;
  private final ImmutableSet<PathFragment> optionalOutputFiles;
  private final RunfilesSupplier runfilesSupplier;
  private final ActionMetadata action;
  private final ResourceSet localResources;

  // TODO(bazel-team): When we migrate ActionSpawn to use this constructor decide on and enforce
  // policy on runfilesManifests and runfilesSupplier being non-empty (ie: are overlapping mappings
  // allowed?).
  @VisibleForTesting
  BaseSpawn(
      List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      Map<PathFragment, Artifact> runfilesManifests,
      RunfilesSupplier runfilesSupplier,
      ActionMetadata action,
      ResourceSet localResources,
      Collection<PathFragment> optionalOutputFiles) {
    this.arguments = ImmutableList.copyOf(arguments);
    this.environment = ImmutableMap.copyOf(environment);
    this.executionInfo = ImmutableMap.copyOf(executionInfo);
    this.runfilesManifests = ImmutableMap.copyOf(runfilesManifests);
    this.runfilesSupplier = runfilesSupplier;
    this.action = action;
    this.localResources = localResources;
    this.optionalOutputFiles = ImmutableSet.copyOf(optionalOutputFiles);
  }

  /**
   * Returns a new Spawn. The caller must not modify the parameters after the call; neither will
   * this method.
   */
  public BaseSpawn(List<String> arguments,
     Map<String, String> environment,
     Map<String, String> executionInfo,
     RunfilesSupplier runfilesSupplier,
     ActionMetadata action,
     ResourceSet localResources) {
    this(
        arguments,
        environment,
        executionInfo,
        ImmutableMap.<PathFragment, Artifact>of(),
        runfilesSupplier,
        action,
        localResources,
        ImmutableSet.<PathFragment>of());
  }

  /**
   * Returns a new Spawn. The caller must not modify the parameters after the call; neither will
   * this method.
   */
  public BaseSpawn(List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      Map<PathFragment, Artifact> runfilesManifests,
      ActionMetadata action,
      ResourceSet localResources) {
    this(
        arguments,
        environment,
        executionInfo,
        runfilesManifests,
        EmptyRunfilesSupplier.INSTANCE,
        action,
        localResources,
        ImmutableSet.<PathFragment>of());
  }

  /**
   * Returns a new Spawn.
   */
  public BaseSpawn(List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      ActionMetadata action,
      ResourceSet localResources) {
    this(
        arguments,
        environment,
        executionInfo,
        ImmutableMap.<PathFragment, Artifact>of(),
        action,
        localResources);
  }

  public BaseSpawn(
      List<String> arguments,
      Map<String, String> environment,
      Map<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier,
      ActionMetadata action,
      ResourceSet localResources,
      Collection<PathFragment> optionalOutputFiles) {
    this(
        arguments,
        environment,
        executionInfo,
        ImmutableMap.<PathFragment, Artifact>of(),
        runfilesSupplier,
        action,
        localResources,
        optionalOutputFiles);
  }

  public static PathFragment runfilesForFragment(PathFragment pathFragment) {
    return pathFragment.getParentDirectory().getChild(pathFragment.getBaseName() + ".runfiles");
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
  public String asShellCommand(Path workingDir) {
    return asShellCommand(getArguments(), workingDir, getEnvironment());
  }

  @Override
  public ImmutableMap<PathFragment, Artifact> getRunfilesManifests() {
    return runfilesManifests;
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
  public SpawnInfo getExtraActionInfo() {
    SpawnInfo.Builder info = SpawnInfo.newBuilder();

    info.addAllArgument(getArguments());
    for (Map.Entry<String, String> variable : getEnvironment().entrySet()) {
      info.addVariable(EnvironmentVariable.newBuilder()
        .setName(variable.getKey())
        .setValue(variable.getValue()).build());
    }
    for (ActionInput input : getInputFiles()) {
      // Explicitly ignore middleman artifacts here.
      if (!(input instanceof Artifact) || !((Artifact) input).isMiddlemanArtifact()) {
        info.addInputFile(input.getExecPathString());
      }
    }
    info.addAllOutputFile(ActionInputHelper.toExecPaths(getOutputFiles()));
    return info.build();
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
    if (runfilesRoot == null) {
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
    if (runfilesSupplierRoots.size() == 1 && runfilesManifests.isEmpty()) {
      return Iterables.getOnlyElement(runfilesSupplierRoots);
    } else if (runfilesManifests.size() == 1 && runfilesSupplierRoots.isEmpty()) {
      return Iterables.getOnlyElement(runfilesManifests.keySet());
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
  public ActionMetadata getResourceOwner() {
    return action;
  }

  @Override
  public ResourceSet getLocalResources() {
    return localResources;
  }

  @Override
  public ActionOwner getOwner() { return action.getOwner(); }

  @Override
  public String getMnemonic() { return action.getMnemonic(); }

  /**
   * Convert a working dir + environment map + arg list into a Bourne shell
   * command.
   */
  public static String asShellCommand(Collection<String> arguments,
                                      Path workingDirectory,
                                      Map<String, String> environment) {
    // We print this command out in such a way that it can safely be
    // copied+pasted as a Bourne shell command.  This is extremely valuable for
    // debugging.
    return CommandFailureUtils.describeCommand(CommandDescriptionForm.COMPLETE,
        arguments, environment, workingDirectory.getPathString());
  }

  /**
   * A local spawn requiring zero resources.
   */
  public static class Local extends BaseSpawn {
    public Local(List<String> arguments, Map<String, String> environment, ActionMetadata action) {
      super(arguments, environment, ImmutableMap.<String, String>of("local", ""),
          action, ResourceSet.ZERO);
    }
  }
}
