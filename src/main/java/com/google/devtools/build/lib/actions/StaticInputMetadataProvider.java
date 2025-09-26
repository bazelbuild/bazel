// Copyright 2019 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/** A {@link InputMetadataProvider} backed by static data */
public final class StaticInputMetadataProvider implements InputMetadataProvider {

  private static final StaticInputMetadataProvider EMPTY =
      new StaticInputMetadataProvider(ImmutableMap.of());

  public static StaticInputMetadataProvider empty() {
    return EMPTY;
  }

  private final ImmutableMap<ActionInput, FileArtifactValue> inputToMetadata;
  private final ImmutableMap<PathFragment, ActionInput> execPathToInput;

  public StaticInputMetadataProvider(Map<ActionInput, FileArtifactValue> inputToMetadata) {
    this.inputToMetadata = ImmutableMap.copyOf(inputToMetadata);
    this.execPathToInput = constructExecPathToInputMap(inputToMetadata.keySet());
  }

  private static ImmutableMap<PathFragment, ActionInput> constructExecPathToInputMap(
      Collection<ActionInput> inputs) {
    ImmutableMap.Builder<PathFragment, ActionInput> builder =
        ImmutableMap.builderWithExpectedSize(inputs.size());
    for (ActionInput input : inputs) {
      builder.put(input.getExecPath(), input);
    }
    return builder.buildOrThrow();
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadataChecked(ActionInput input) {
    return inputToMetadata.get(input);
  }

  @Nullable
  @Override
  public TreeArtifactValue getTreeMetadata(ActionInput actionInput) {
    return null;
  }

  @Nullable
  @Override
  public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
    return null;
  }

  @Override
  @Nullable
  public FilesetOutputTree getFileset(ActionInput input) {
    return null;
  }

  @Override
  public ImmutableMap<Artifact, FilesetOutputTree> getFilesets() {
    return ImmutableMap.of();
  }

  @Override
  @Nullable
  public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
    return null;
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public ActionInput getInput(PathFragment execPath) {
    return execPathToInput.get(execPath);
  }
}
