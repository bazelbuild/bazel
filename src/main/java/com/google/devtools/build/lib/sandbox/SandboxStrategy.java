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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSet.Builder;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/** Abstract common ancestor for sandbox strategies implementing the common parts. */
abstract class SandboxStrategy implements SpawnActionContext {

  private final BlazeDirectories blazeDirs;
  private final Path execRoot;
  private final boolean verboseFailures;
  private final SandboxOptions sandboxOptions;

  public SandboxStrategy(
      BlazeDirectories blazeDirs, boolean verboseFailures, SandboxOptions sandboxOptions) {
    this.blazeDirs = blazeDirs;
    this.execRoot = blazeDirs.getExecRoot();
    this.verboseFailures = verboseFailures;
    this.sandboxOptions = sandboxOptions;
  }

  /** Gets the list of directories that the spawn will assume to be writable. */
  protected ImmutableSet<Path> getWritableDirs(Path sandboxExecRoot, Map<String, String> env) {
    Builder<Path> writableDirs = ImmutableSet.builder();
    // We have to make the TEST_TMPDIR directory writable if it is specified.
    if (env.containsKey("TEST_TMPDIR")) {
      writableDirs.add(sandboxExecRoot.getRelative(env.get("TEST_TMPDIR")));
    }
    return writableDirs.build();
  }

  protected ImmutableSet<Path> getInaccessiblePaths() {
    ImmutableSet.Builder<Path> inaccessiblePaths = ImmutableSet.builder();
    for (String path : sandboxOptions.sandboxBlockPath) {
      inaccessiblePaths.add(blazeDirs.getFileSystem().getPath(path));
    }
    return inaccessiblePaths.build();
  }

  /** Mount all runfiles that the spawn needs as specified in its runfiles manifests. */
  protected void mountRunfilesFromManifests(Map<PathFragment, Path> mounts, Spawn spawn)
      throws IOException, ExecException {
    for (Map.Entry<PathFragment, Artifact> manifest : spawn.getRunfilesManifests().entrySet()) {
      String manifestFilePath = manifest.getValue().getPath().getPathString();
      Preconditions.checkState(!manifest.getKey().isAbsolute());
      PathFragment targetDirectory = manifest.getKey();

      parseManifestFile(
          blazeDirs.getFileSystem(),
          mounts,
          targetDirectory,
          new File(manifestFilePath),
          false,
          "");
    }
  }

  /** Mount all files that the spawn needs as specified in its fileset manifests. */
  protected void mountFilesFromFilesetManifests(
      Map<PathFragment, Path> mounts, Spawn spawn, ActionExecutionContext executionContext)
      throws IOException, ExecException {
    final FilesetActionContext filesetContext =
        executionContext.getExecutor().getContext(FilesetActionContext.class);
    for (Artifact fileset : spawn.getFilesetManifests()) {
      File manifestFile =
          new File(
              execRoot.getPathString(),
              AnalysisUtils.getManifestPathFromFilesetPath(fileset.getExecPath()).getPathString());
      PathFragment targetDirectory = fileset.getExecPath();

      parseManifestFile(
          blazeDirs.getFileSystem(),
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
      throws IOException, ExecException {
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
        PathFragment targetPathFragment = new PathFragment(fields[0]);
        if (!workspaceName.isEmpty()) {
          if (!targetPathFragment.getSegment(0).equals(workspaceName)) {
            throw new EnvironmentalExecException(
                "Fileset manifest line must start with workspace name");
          }
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
          source = fs.getPath("/dev/null");
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
  protected void mountRunfilesFromSuppliers(Map<PathFragment, Path> mounts, Spawn spawn)
      throws IOException {
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings =
        spawn.getRunfilesSupplier().getMappings();
    for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root = rootAndMappings.getKey();
      if (root.isAbsolute()) {
        root = root.relativeTo(execRoot.asFragment());
      }
      for (Map.Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        Artifact sourceArtifact = mapping.getValue();
        PathFragment source =
            (sourceArtifact != null) ? sourceArtifact.getExecPath() : new PathFragment("/dev/null");

        Preconditions.checkArgument(!mapping.getKey().isAbsolute());
        PathFragment target = root.getRelative(mapping.getKey());
        mounts.put(target, execRoot.getRelative(source));
      }
    }
  }

  /** Mount all inputs of the spawn. */
  protected void mountInputs(
      Map<PathFragment, Path> mounts, Spawn spawn, ActionExecutionContext actionExecutionContext) {
    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(
            spawn.getInputFiles(), actionExecutionContext.getArtifactExpander());

    if (spawn.getResourceOwner() instanceof CppCompileAction) {
      CppCompileAction action = (CppCompileAction) spawn.getResourceOwner();
      if (action.shouldScanIncludes()) {
        inputs.addAll(action.getAdditionalInputs());
      }
    }

    for (ActionInput input : inputs) {
      if (input.getExecPathString().contains("internal/_middlemen/")) {
        continue;
      }
      PathFragment mount = new PathFragment(input.getExecPathString());
      mounts.put(mount, execRoot.getRelative(mount));
    }
  }

  @Override
  public boolean willExecuteRemotely(boolean remotable) {
    return false;
  }

  @Override
  public String toString() {
    return "sandboxed";
  }

  @Override
  public boolean shouldPropagateExecException() {
    return verboseFailures && sandboxOptions.sandboxDebug;
  }
}
