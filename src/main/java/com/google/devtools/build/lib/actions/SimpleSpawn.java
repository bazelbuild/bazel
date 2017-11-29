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
import javax.annotation.concurrent.Immutable;

/**
 * Immutable implementation of a Spawn that does not perform any processing on the parameters.
 * Prefer this over all other Spawn implementations.
 */
@Immutable
public final class SimpleSpawn implements Spawn {
  private final ActionExecutionMetadata owner;
  private final ImmutableList<String> arguments;
  private final ImmutableMap<String, String> environment;
  private final ImmutableMap<String, String> executionInfo;
  private final ImmutableList<? extends ActionInput> inputs;
  private final ImmutableList<? extends ActionInput> tools;
  private final RunfilesSupplier runfilesSupplier;
  private final ImmutableList<Artifact> filesetManifests;
  private final ImmutableList<? extends ActionInput> outputs;
  private final ResourceSet localResources;

  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier,
      ImmutableList<? extends ActionInput> inputs,
      ImmutableList<? extends ActionInput> tools,
      ImmutableList<Artifact> filesetManifests,
      ImmutableList<? extends ActionInput> outputs,
      ResourceSet localResources) {
    this.owner = Preconditions.checkNotNull(owner);
    this.arguments = Preconditions.checkNotNull(arguments);
    this.environment = Preconditions.checkNotNull(environment);
    this.executionInfo = Preconditions.checkNotNull(executionInfo);
    this.inputs = Preconditions.checkNotNull(inputs);
    this.tools = Preconditions.checkNotNull(tools);
    this.runfilesSupplier =
        runfilesSupplier == null ? EmptyRunfilesSupplier.INSTANCE : runfilesSupplier;
    this.filesetManifests = Preconditions.checkNotNull(filesetManifests);
    this.outputs = Preconditions.checkNotNull(outputs);
    this.localResources = Preconditions.checkNotNull(localResources);
  }

  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      ImmutableList<? extends ActionInput> inputs,
      ImmutableList<? extends ActionInput> outputs,
      ResourceSet localResources) {
    this(
        owner,
        arguments,
        environment,
        executionInfo,
        null,
        inputs,
        ImmutableList.<Artifact>of(),
        ImmutableList.<Artifact>of(),
        outputs,
        localResources);
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
    return filesetManifests;
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
  public ImmutableList<? extends ActionInput> getInputFiles() {
    return inputs;
  }

  @Override
  public ImmutableList<? extends ActionInput> getToolFiles() {
    return tools;
  }

  @Override
  public ImmutableList<? extends ActionInput> getOutputFiles() {
    return outputs;
  }

  @Override
  public ActionExecutionMetadata getResourceOwner() {
    return owner;
  }

  @Override
  public ResourceSet getLocalResources() {
    return localResources;
  }

  @Override
  public String getMnemonic() {
    return owner.getMnemonic();
  }
}
