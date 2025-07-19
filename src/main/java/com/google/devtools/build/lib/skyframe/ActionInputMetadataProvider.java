// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * This class stores the metadata for the inputs of an action.
 *
 * <p>It is constructed during the preparation for the execution of the action and garbage collected
 * once the action finishes executing.
 */
public final class ActionInputMetadataProvider implements InputMetadataProvider {

  private final ActionInputMap inputArtifactData;

  /**
   * Supports looking up a {@link FilesetOutputSymlink} by the target's exec path.
   *
   * <p>Initialized lazily because it can consume significant memory and may never be needed, for
   * example if there is an action cache hit.
   */
  private final Supplier<ImmutableMap<String, FilesetOutputSymlink>> filesetMapping;

  public ActionInputMetadataProvider(ActionInputMap inputArtifactData) {
    this.inputArtifactData = inputArtifactData;
    this.filesetMapping =
        Suppliers.memoize(() -> createFilesetMapping(inputArtifactData.getFilesets()));
  }

  private static ImmutableMap<String, FilesetOutputSymlink> createFilesetMapping(
      Map<Artifact, FilesetOutputTree> filesets) {
    Map<String, FilesetOutputSymlink> filesetMap = new HashMap<>();
    for (FilesetOutputTree filesetOutput : filesets.values()) {
      for (FilesetOutputSymlink link : filesetOutput.symlinks()) {
        filesetMap.put(link.target().getExecPathString(), link);
      }
    }
    return ImmutableMap.copyOf(filesetMap);
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadataChecked(ActionInput actionInput) throws IOException {
    if (!(actionInput instanceof Artifact artifact)) {
      return null;
    }
    FileArtifactValue value = inputArtifactData.getInputMetadataChecked(artifact);
    if (value != null) {
      return checkExists(value, artifact);
    }
    FilesetOutputSymlink filesetLink = filesetMapping.get().get(artifact.getExecPathString());
    if (filesetLink != null) {
      return filesetLink.metadata();
    }
    return null;
  }

  @Nullable
  @Override
  public TreeArtifactValue getTreeMetadata(ActionInput actionInput) {
    return inputArtifactData.getTreeMetadata(actionInput);
  }

  @Nullable
  @Override
  public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
    return inputArtifactData.getEnclosingTreeMetadata(execPath);
  }

  @Nullable
  @Override
  public FilesetOutputTree getFileset(ActionInput input) {
    return inputArtifactData.getFileset(input);
  }

  @Override
  public Map<Artifact, FilesetOutputTree> getFilesets() {
    return inputArtifactData.getFilesets();
  }

  @Nullable
  @Override
  public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
    return inputArtifactData.getRunfilesMetadata(input);
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    return inputArtifactData.getRunfilesTrees();
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPath) {
    ActionInput input = inputArtifactData.getInput(execPath);
    if (input != null) {
      return input;
    }
    FilesetOutputSymlink filesetLink = filesetMapping.get().get(execPath);
    if (filesetLink != null) {
      return filesetLink.target();
    }
    return null;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("inputArtifactDataSize", inputArtifactData.sizeForDebugging())
        .toString();
  }

  /**
   * If {@code value} represents an existing file, returns it as is, otherwise throws {@link
   * FileNotFoundException}.
   */
  private static FileArtifactValue checkExists(FileArtifactValue value, Artifact artifact)
      throws FileNotFoundException {
    if (FileArtifactValue.MISSING_FILE_MARKER.equals(value)) {
      throw new FileNotFoundException(artifact + " does not exist");
    }
    return checkNotNull(value, artifact);
  }
}
