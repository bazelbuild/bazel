// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.OS;
import java.util.Collection;
import java.util.Set;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/** Immutable implementation of a Spawn that does not perform any processing on the parameters. */
@Immutable
public final class SimpleSpawn implements Spawn {
  private final ActionExecutionMetadata owner;
  private final ImmutableList<String> arguments;
  private final ImmutableMap<String, String> environment;
  private final ImmutableMap<String, String> executionInfo;
  private final SpawnInputs inputs;
  private final NestedSet<? extends ActionInput> tools;
  private final ImmutableList<ActionInput> outputs;
  // If null, all outputs are mandatory.
  @Nullable private final Set<? extends ActionInput> mandatoryOutputs;
  private final PathMapper pathMapper;
  private final ResourceSetOrBuilder localResources;
  @Nullable private ResourceSet localResourcesCached;

  @SuppressWarnings("TooManyParameters")
  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      SpawnInputs inputs,
      NestedSet<? extends ActionInput> tools,
      Collection<? extends ActionInput> outputs,
      @Nullable Set<? extends ActionInput> mandatoryOutputs,
      ResourceSetOrBuilder localResources,
      PathMapper pathMapper) {
    this.owner = Preconditions.checkNotNull(owner);
    this.arguments = Preconditions.checkNotNull(arguments);
    this.environment = Preconditions.checkNotNull(environment);
    this.executionInfo = Preconditions.checkNotNull(executionInfo);
    this.inputs = Preconditions.checkNotNull(inputs);
    this.tools = Preconditions.checkNotNull(tools);
    this.outputs = ImmutableList.copyOf(outputs);
    this.mandatoryOutputs = mandatoryOutputs;
    this.localResources = Preconditions.checkNotNull(localResources);
    this.localResourcesCached = null;
    this.pathMapper = pathMapper;
  }

  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      SpawnInputs inputs,
      NestedSet<? extends ActionInput> tools,
      Collection<? extends ActionInput> outputs,
      @Nullable Set<? extends ActionInput> mandatoryOutputs,
      ResourceSetOrBuilder localResources) {
    this(
        owner,
        arguments,
        environment,
        executionInfo,
        inputs,
        tools,
        outputs,
        mandatoryOutputs,
        localResources,
        PathMapper.NOOP);
  }

  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      NestedSet<? extends ActionInput> inputs,
      Collection<? extends ActionInput> outputs,
      ResourceSetOrBuilder localResources) {
    this(
        owner,
        arguments,
        environment,
        executionInfo,
        SpawnInputs.of(inputs),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        outputs,
        /* mandatoryOutputs= */ null,
        localResources);
  }

  @Override
  public ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public ImmutableList<String> getArguments() {
    return arguments;
  }

  @Override
  public ImmutableMap<String, String> getEnvironment() {
    return environment;
  }

  @Override
  public SpawnInputs getInputFiles() {
    return inputs;
  }

  @Override
  public NestedSet<? extends ActionInput> getToolFiles() {
    return tools;
  }

  @Override
  public ImmutableList<ActionInput> getOutputFiles() {
    return outputs;
  }

  @Override
  public boolean isMandatoryOutput(ActionInput output) {
    return mandatoryOutputs == null || mandatoryOutputs.contains(output);
  }

  @Override
  public ActionExecutionMetadata getResourceOwner() {
    return owner;
  }

  @Override
  public ResourceSet getLocalResources() throws ExecException, InterruptedException {
    ResourceSet result = localResourcesCached;
    if (result == null) {
      // Not expected to be called concurrently, and an idempotent computation if it is.
      result =
          localResources.buildLocalResources(
              OS.getCurrent(),
              inputs.flatten().size(),
              getExecutionInfo(),
              getCombinedExecProperties());
      localResourcesCached = result;
    }
    return result;
  }

  @Override
  public PathMapper getPathMapper() {
    return pathMapper;
  }

  @Override
  public String getMnemonic() {
    return owner.getMnemonic();
  }

  @Override
  @Nullable
  public PlatformInfo getExecutionPlatform() {
    return owner.getExecutionPlatform();
  }

  @Override
  public String toString() {
    return Spawns.prettyPrint(this);
  }
}
