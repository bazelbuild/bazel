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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import javax.annotation.Nullable;
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
  private final NestedSet<? extends ActionInput> inputs;
  private final NestedSet<? extends ActionInput> tools;
  private final RunfilesSupplier runfilesSupplier;
  private final ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings;
  private final ImmutableList<? extends ActionInput> outputs;
  private final ResourceSet localResources;

  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      RunfilesSupplier runfilesSupplier,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings,
      NestedSet<? extends ActionInput> inputs,
      NestedSet<? extends ActionInput> tools,
      ImmutableSet<? extends ActionInput> outputs,
      ResourceSet localResources) {
    this.owner = Preconditions.checkNotNull(owner);
    this.arguments = Preconditions.checkNotNull(arguments);
    this.environment = Preconditions.checkNotNull(environment);
    this.executionInfo = Preconditions.checkNotNull(executionInfo);
    this.inputs = Preconditions.checkNotNull(inputs);
    this.tools = Preconditions.checkNotNull(tools);
    this.runfilesSupplier =
        runfilesSupplier == null ? EmptyRunfilesSupplier.INSTANCE : runfilesSupplier;
    this.filesetMappings = filesetMappings;
    this.outputs = Preconditions.checkNotNull(outputs).asList();
    this.localResources = Preconditions.checkNotNull(localResources);
  }

  public SimpleSpawn(
      ActionExecutionMetadata owner,
      ImmutableList<String> arguments,
      ImmutableMap<String, String> environment,
      ImmutableMap<String, String> executionInfo,
      NestedSet<? extends ActionInput> inputs,
      ImmutableSet<? extends ActionInput> outputs,
      ResourceSet localResources) {
    this(
        owner,
        arguments,
        environment,
        executionInfo,
        null,
        ImmutableMap.of(),
        inputs,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        outputs,
        localResources);
  }

  @Override
  public ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public RunfilesSupplier getRunfilesSupplier() {
    return runfilesSupplier;
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
  public ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> getFilesetMappings() {
    return filesetMappings;
  }

  @Override
  public NestedSet<? extends ActionInput> getInputFiles() {
    return inputs;
  }

  @Override
  public NestedSet<? extends ActionInput> getToolFiles() {
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

  @Override
  public ImmutableMap<String, String> getCombinedExecProperties() {
    return owner.getExecProperties();
  }

  @Override
  @Nullable
  public PlatformInfo getExecutionPlatform() {
    return owner.getExecutionPlatform();
  }
}
