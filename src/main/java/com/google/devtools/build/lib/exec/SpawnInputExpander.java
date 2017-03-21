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
package com.google.devtools.build.lib.exec;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.io.LineProcessor;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * A helper class for spawn strategies to turn runfiles suppliers into input mappings. This class
 * performs no I/O operations, but only rearranges the files according to how the runfiles should be
 * laid out.
 */
public class SpawnInputExpander {
  public static final ActionInput EMPTY_FILE = null;

  private final boolean strict;

  /**
   * Creates a new instance. If strict is true, then the expander checks for directories in runfiles
   * and throws an exception if it finds any. Otherwise it silently ignores directories in runfiles
   * and adds a mapping for them. At this time, directories in filesets are always silently added
   * as mappings.
   *
   * <p>Directories in inputs are a correctness issue: Bazel only tracks dependencies at the action
   * level, and it does not track dependencies on directories. Making a directory available to a
   * spawn even though it's contents are not tracked as dependencies leads to incorrect incremental
   * builds, since changes to the contents do not trigger action invalidation.
   *
   * <p>As such, all spawn strategies should always be strict and not make directories available to
   * the subprocess. However, that's a breaking change, and therefore we make it depend on this flag
   * for now.
   */
  public SpawnInputExpander(boolean strict) {
    this.strict = strict;
  }

  private void addMapping(
      Map<PathFragment, ActionInput> inputMappings,
      PathFragment targetLocation,
      ActionInput input) {
    Preconditions.checkArgument(!targetLocation.isAbsolute(), targetLocation);
    if (!inputMappings.containsKey(targetLocation)) {
      inputMappings.put(targetLocation, input);
    }
  }

  /** Adds runfiles inputs from runfilesSupplier to inputMappings. */
  @VisibleForTesting
  void addRunfilesToInputs(
      Map<PathFragment, ActionInput> inputMap,
      RunfilesSupplier runfilesSupplier,
      ActionInputFileCache actionFileCache) throws IOException {
    Map<PathFragment, Map<PathFragment, Artifact>> rootsAndMappings = null;
    rootsAndMappings = runfilesSupplier.getMappings();

    for (Entry<PathFragment, Map<PathFragment, Artifact>> rootAndMappings :
        rootsAndMappings.entrySet()) {
      PathFragment root = rootAndMappings.getKey();
      Preconditions.checkState(!root.isAbsolute(), root);
      for (Entry<PathFragment, Artifact> mapping : rootAndMappings.getValue().entrySet()) {
        PathFragment location = root.getRelative(mapping.getKey());
        Artifact localArtifact = mapping.getValue();
        if (localArtifact != null) {
          if (strict && !actionFileCache.isFile(localArtifact)) {
            throw new IOException("Not a file: " + localArtifact.getPath().getPathString());
          }
          addMapping(inputMap, location, localArtifact);
        } else {
          addMapping(inputMap, location, EMPTY_FILE);
        }
      }
    }
  }

  /**
   * Parses the fileset manifest file, adding to the inputMappings where
   * appropriate. Lines referring to directories are recursed.
   */
  @VisibleForTesting
  void parseFilesetManifest(
      Map<PathFragment, ActionInput> inputMappings, Artifact manifest, String workspaceName)
          throws IOException {
    Path file = manifest.getRoot().getPath().getRelative(
        AnalysisUtils.getManifestPathFromFilesetPath(manifest.getExecPath()).getPathString());
    FileSystemUtils.asByteSource(file).asCharSource(UTF_8)
        .readLines(new ManifestLineProcessor(inputMappings, workspaceName, manifest.getExecPath()));
  }

  private final class ManifestLineProcessor implements LineProcessor<Object> {
    private final Map<PathFragment, ActionInput> inputMap;
    private final String workspaceName;
    private final PathFragment targetPrefix;
    private int lineNum = 0;

    ManifestLineProcessor(
        Map<PathFragment, ActionInput> inputMap,
        String workspaceName,
        PathFragment targetPrefix) {
      this.inputMap = inputMap;
      this.workspaceName = workspaceName;
      this.targetPrefix = targetPrefix;
    }

    @Override
    public boolean processLine(String line) throws IOException {
      if (++lineNum % 2 == 0) {
        // Digest line, skip.
        return true;
      }
      if (line.isEmpty()) {
        return true;
      }

      ActionInput artifact;
      PathFragment location;
      int pos = line.indexOf(' ');
      if (pos == -1) {
        location = new PathFragment(line);
        artifact = EMPTY_FILE;
      } else {
        String targetPath = line.substring(pos + 1);
        if (targetPath.charAt(0) != '/') {
          throw new IOException(String.format("runfiles target is not absolute: %s", targetPath));
        }
        artifact = targetPath.isEmpty() ? EMPTY_FILE : ActionInputHelper.fromPath(targetPath);

        location = new PathFragment(line.substring(0, pos));
        if (!workspaceName.isEmpty()) {
          if (!location.getSegment(0).equals(workspaceName)) {
            throw new IOException(
                String.format(
                    "fileset manifest line must start with '%s': '%s'", workspaceName, location));
          } else {
            // Erase "<workspaceName>/".
            location = location.subFragment(1, location.segmentCount());
          }
        }
      }

      addMapping(inputMap, targetPrefix.getRelative(location), artifact);
      return true;
    }

    @Override
    public Object getResult() {
      return null; // Unused.
    }
  }

  private void addInputs(
      Map<PathFragment, ActionInput> inputMap, Spawn spawn, ArtifactExpander artifactExpander) {
    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(spawn.getInputFiles(), artifactExpander);
    for (ActionInput input : inputs) {
      addMapping(inputMap, input.getExecPath(), input);
    }
  }

  /**
   * Convert the inputs of the given spawn to a map from exec-root relative paths to action inputs.
   * In some cases, this generates empty files, for which it uses {@code null}.
   */
  public SortedMap<PathFragment, ActionInput> getInputMapping(
      Spawn spawn, ArtifactExpander artifactExpander, ActionInputFileCache actionInputFileCache,
      FilesetActionContext filesetContext)
          throws IOException {
    TreeMap<PathFragment, ActionInput> inputMap = new TreeMap<>();
    addInputs(inputMap, spawn, artifactExpander);
    addRunfilesToInputs(
        inputMap, spawn.getRunfilesSupplier(), actionInputFileCache);
    for (Artifact manifest : spawn.getFilesetManifests()) {
      parseFilesetManifest(inputMap, manifest, filesetContext.getWorkspaceName());
    }
    return inputMap;
  }
}
