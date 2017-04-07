// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Contains common helper methods that extract information from {@link Spawn} objects. */
public final class SpawnHelpers {

  private final Path execRoot;

  public SpawnHelpers(Path execRoot) {
    this.execRoot = execRoot;
  }

  /**
   * Returns the inputs of a Spawn as a map of PathFragments relative to an execRoot to paths in the
   * host filesystem where the input files can be found.
   */
  public Map<PathFragment, Path> getMounts(Spawn spawn, ActionExecutionContext executionContext)
      throws IOException {
    Map<PathFragment, Path> mounts = new HashMap<>();
    mountRunfilesFromSuppliers(mounts, spawn);
    mountFilesFromFilesetManifests(mounts, spawn, executionContext);
    mountInputs(mounts, spawn, executionContext);
    return mounts;
  }

  /** Mount all files that the spawn needs as specified in its fileset manifests. */
  void mountFilesFromFilesetManifests(
      Map<PathFragment, Path> mounts, Spawn spawn, ActionExecutionContext executionContext)
      throws IOException {
    final FilesetActionContext filesetContext =
        executionContext.getExecutor().getContext(FilesetActionContext.class);
    for (Artifact fileset : spawn.getFilesetManifests()) {
      File manifestFile =
          new File(
              execRoot.getPathString(),
              AnalysisUtils.getManifestPathFromFilesetPath(fileset.getExecPath()).getPathString());
      PathFragment targetDirectory = fileset.getExecPath();

      parseManifestFile(
          execRoot.getFileSystem(),
          mounts,
          targetDirectory,
          manifestFile,
          true,
          filesetContext.getWorkspaceName());
    }
  }

  /** A parser for the MANIFEST files used by Filesets and runfiles. */
  static void parseManifestFile(
      FileSystem fs,
      Map<PathFragment, Path> mounts,
      PathFragment targetDirectory,
      File manifestFile,
      boolean isFilesetManifest,
      String workspaceName)
      throws IOException {
    int lineNum = 0;
    for (String line : Files.readLines(manifestFile, StandardCharsets.UTF_8)) {
      if (isFilesetManifest && (++lineNum % 2 == 0)) {
        continue;
      }
      if (line.isEmpty()) {
        continue;
      }

      String[] fields = line.trim().split(" ");

      // The "target" field is always a relative path that is to be interpreted in this way:
      // (1) If this is a fileset manifest and our workspace name is not empty, the first segment
      // of each "target" path must be the workspace name, which is then stripped before further
      // processing.
      // (2) The "target" path is then appended to the "targetDirectory", which is a path relative
      // to the execRoot. Together, this results in the full path in the execRoot in which place a
      // symlink referring to "source" has to be created (see below).
      PathFragment targetPath;
      if (isFilesetManifest) {
        PathFragment targetPathFragment = PathFragment.create(fields[0]);
        if (!workspaceName.isEmpty()) {
          Preconditions.checkState(
              targetPathFragment.getSegment(0).equals(workspaceName),
              "Fileset manifest line must start with workspace name");
          targetPathFragment = targetPathFragment.subFragment(1, targetPathFragment.segmentCount());
        }
        targetPath = targetDirectory.getRelative(targetPathFragment);
      } else {
        targetPath = targetDirectory.getRelative(fields[0]);
      }

      // The "source" field, if it exists, is always an absolute path and may point to any file in
      // the filesystem (it is not limited to files in the workspace or execroot).
      Path source;
      switch (fields.length) {
        case 1:
          source = null;
          break;
        case 2:
          source = fs.getPath(fields[1]);
          break;
        default:
          throw new IllegalStateException("'" + line + "' splits into more than 2 parts");
      }

      mounts.put(targetPath, source);
    }
  }

  /** Mount all runfiles that the spawn needs as specified via its runfiles suppliers. */
  void mountRunfilesFromSuppliers(Map<PathFragment, Path> mounts, Spawn spawn) throws IOException {
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        spawn.getRunfilesSupplier().getMappings();
    for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root = rootAndMappings.getKey();
      Preconditions.checkState(!root.isAbsolute());
      for (Map.Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        Artifact sourceArtifact = mapping.getValue();
        Path source =
            (sourceArtifact != null) ? execRoot.getRelative(sourceArtifact.getExecPath()) : null;

        Preconditions.checkArgument(!mapping.getKey().isAbsolute());
        PathFragment target = root.getRelative(mapping.getKey());
        mounts.put(target, source);
      }
    }
  }

  /** Mount all inputs of the spawn. */
  void mountInputs(
      Map<PathFragment, Path> mounts, Spawn spawn, ActionExecutionContext actionExecutionContext) {
    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(
            spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());

    // ActionInputHelper#expandArtifacts above expands empty TreeArtifacts into an empty list.
    // However, actions that accept TreeArtifacts as inputs generally expect that the empty
    // directory is created. So here we explicitly mount the directories of the TreeArtifacts as
    // inputs.
    for (ActionInput input : spawn.getInputFiles()) {
      if (input instanceof Artifact && ((Artifact) input).isTreeArtifact()) {
        List<Artifact> containedArtifacts = new ArrayList<>();
        actionExecutionContext.getArtifactExpander().expand((Artifact) input, containedArtifacts);
        // Attempting to mount a non-empty directory results in ERR_DIRECTORY_NOT_EMPTY, so we only
        // mount empty TreeArtifacts as directories.
        if (containedArtifacts.isEmpty()) {
          PathFragment mount = PathFragment.create(input.getExecPathString());
          mounts.put(mount, execRoot.getRelative(mount));
        }
      }
    }

    for (ActionInput input : inputs) {
      if (input.getExecPathString().contains("internal/_middlemen/")) {
        continue;
      }
      PathFragment mount = PathFragment.create(input.getExecPathString());
      mounts.put(mount, execRoot.getRelative(mount));
    }
  }
}
