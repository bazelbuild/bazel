// Copyright 2017 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec.util;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Builder class to create {@link Spawn} instances for testing.
 */
public final class SpawnBuilder {
  private String mnemonic = "Mnemonic";
  private String progressMessage = "progress message";
  private String ownerLabel = "//dummy:label";
  @Nullable private Artifact ownerPrimaryOutput;
  @Nullable private PlatformInfo platform;
  private final List<String> args;
  private final Map<String, String> environment = new HashMap<>();
  private final Map<String, String> executionInfo = new HashMap<>();
  private ImmutableMap<String, String> execProperties = ImmutableMap.of();
  private final NestedSetBuilder<ActionInput> inputs = NestedSetBuilder.stableOrder();
  private final List<ActionInput> outputs = new ArrayList<>();
  private final Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesetMappings =
      new HashMap<>();
  private final NestedSetBuilder<ActionInput> tools = NestedSetBuilder.stableOrder();

  private RunfilesSupplier runfilesSupplier = EmptyRunfilesSupplier.INSTANCE;
  private ResourceSet resourceSet = ResourceSet.ZERO;

  public SpawnBuilder(String... args) {
    this.args = ImmutableList.copyOf(args);
  }

  public Spawn build() {
    ActionExecutionMetadata owner =
        new FakeOwner(
            mnemonic, progressMessage, ownerLabel, ownerPrimaryOutput, platform, execProperties);
    return new SimpleSpawn(
        owner,
        ImmutableList.copyOf(args),
        ImmutableMap.copyOf(environment),
        ImmutableMap.copyOf(executionInfo),
        runfilesSupplier,
        ImmutableMap.copyOf(filesetMappings),
        inputs.build(),
        tools.build(),
        ImmutableSet.copyOf(outputs),
        resourceSet);
  }

  public SpawnBuilder withPlatform(PlatformInfo platform) {
    this.platform = platform;
    return this;
  }

  public SpawnBuilder withMnemonic(String mnemonic) {
    this.mnemonic = checkNotNull(mnemonic);
    return this;
  }

  public SpawnBuilder withProgressMessage(String progressMessage) {
    this.progressMessage = progressMessage;
    return this;
  }

  public SpawnBuilder withOwnerLabel(String ownerLabel) {
    this.ownerLabel = checkNotNull(ownerLabel);
    return this;
  }

  public SpawnBuilder withOwnerPrimaryOutput(Artifact output) {
    ownerPrimaryOutput = checkNotNull(output);
    return this;
  }

  public SpawnBuilder withEnvironment(String key, String value) {
    this.environment.put(key, value);
    return this;
  }

  public SpawnBuilder withExecutionInfo(String key, String value) {
    this.executionInfo.put(key, value);
    return this;
  }

  public SpawnBuilder withExecProperties(ImmutableMap<String, String> execProperties) {
    this.execProperties = execProperties;
    return this;
  }

  public SpawnBuilder withInput(ActionInput input) {
    this.inputs.add(input);
    return this;
  }

  public SpawnBuilder withInput(String name) {
    this.inputs.add(ActionInputHelper.fromPath(name));
    return this;
  }

  public SpawnBuilder withInputs(String... names) {
    for (String name : names) {
      this.inputs.add(ActionInputHelper.fromPath(name));
    }
    return this;
  }

  public SpawnBuilder withOutput(ActionInput output) {
    outputs.add(output);
    return this;
  }

  public SpawnBuilder withOutput(String name) {
    return withOutput(ActionInputHelper.fromPath(name));
  }

  public SpawnBuilder withOutputs(ActionInput... outputs) {
    for (ActionInput output : outputs) {
      withOutput(output);
    }
    return this;
  }

  public SpawnBuilder withOutputs(String... names) {
    for (String name : names) {
      this.outputs.add(ActionInputHelper.fromPath(name));
    }
    return this;
  }

  public SpawnBuilder withFilesetMapping(
      Artifact fileset, ImmutableList<FilesetOutputSymlink> mappings) {
    Preconditions.checkArgument(fileset.isFileset(), "Artifact %s is not fileset", fileset);
    filesetMappings.put(fileset, mappings);
    return this;
  }

  public SpawnBuilder withRunfilesSupplier(RunfilesSupplier runfilesSupplier) {
    this.runfilesSupplier = runfilesSupplier;
    return this;
  }

  public SpawnBuilder withTool(ActionInput tool) {
    tools.add(tool);
    return this;
  }

  public SpawnBuilder withLocalResources(ResourceSet resourceSet) {
    this.resourceSet = resourceSet;
    return this;
  }
}
