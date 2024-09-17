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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetManifest;
import com.google.devtools.build.lib.actions.FilesetManifest.RelativeSymlinkBehaviorWithoutError;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * This class stores the metadata for the inputs of an action.
 *
 * <p>It is constructed during the preparation for the execution of the action and garbage collected
 * once the action finishes executing.
 */
final class ActionInputMetadataProvider implements InputMetadataProvider {
  private final PathFragment execRoot;

  private final ActionInputMap inputArtifactData;

  /** Mapping from a fileset entry's target path to its metadata. */
  private final ImmutableMap<String, FileArtifactValue> filesetMapping;

  ActionInputMetadataProvider(
      PathFragment execRoot,
      ActionInputMap inputArtifactData,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {
    this.execRoot = execRoot;
    this.inputArtifactData = inputArtifactData;
    this.filesetMapping = createFilesetMapping(filesets, execRoot);
  }

  private static ImmutableMap<String, FileArtifactValue> createFilesetMapping(
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets, PathFragment execRoot) {
    Map<String, FileArtifactValue> filesetMap = new HashMap<>();
    for (ImmutableList<FilesetOutputSymlink> links : filesets.values()) {
      FilesetManifest manifest =
          FilesetManifest.constructFilesetManifestWithoutError(
              links, execRoot, RelativeSymlinkBehaviorWithoutError.RESOLVE);
      manifest
          .getArtifactValues()
          .forEach(
              (targetPath, metadata) -> {
                if (metadata.getDigest() != null) {
                  filesetMap.put(targetPath, metadata);
                }
              });
    }
    return ImmutableMap.copyOf(filesetMap);
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadata(ActionInput actionInput) throws IOException {
    if (!(actionInput instanceof Artifact artifact)) {
      PathFragment inputPath = actionInput.getExecPath();
      PathFragment filesetKeyPath =
          inputPath.startsWith(execRoot) ? inputPath.relativeTo(execRoot) : inputPath;
      return filesetMapping.get(filesetKeyPath.getPathString());
    }

    FileArtifactValue value;

    value = inputArtifactData.getInputMetadata(artifact);
    if (value != null) {
      return checkExists(value, artifact);
    }

    return null;
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

  @Override
  public ActionInput getInput(String execPath) {
    return inputArtifactData.getInput(execPath);
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
    if (FileArtifactValue.MISSING_FILE_MARKER.equals(value)
        || FileArtifactValue.OMITTED_FILE_MARKER.equals(value)) {
      throw new FileNotFoundException(artifact + " does not exist");
    }
    return checkNotNull(value, artifact);
  }
}
