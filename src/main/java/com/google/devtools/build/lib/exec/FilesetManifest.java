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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.LineProcessor;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.IORuntimeException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Representation of a Fileset manifest.
 */
public final class FilesetManifest {
  private static final Logger logger = Logger.getLogger(FilesetManifest.class.getName());

  /**
   * Mode that determines how to handle relative target paths.
   */
  public enum RelativeSymlinkBehavior {
    /** Ignore any relative target paths. */
    IGNORE,

    /** Give an error if a relative target path is encountered. */
    ERROR,

    /** Resolve all relative target paths. */
    RESOLVE;
  }

  public static FilesetManifest parseManifestFile(
      PathFragment manifest,
      Path execRoot,
      String workspaceName,
      RelativeSymlinkBehavior relSymlinkBehavior)
      throws IOException {
    Path file = execRoot.getRelative(AnalysisUtils.getManifestPathFromFilesetPath(manifest));
    try {
      return FileSystemUtils.asByteSource(file).asCharSource(UTF_8)
          .readLines(
              new ManifestLineProcessor(workspaceName, manifest, relSymlinkBehavior));
    } catch (IORuntimeException e) {
      // We can't throw IOException from getResult below, so we instead use an unchecked exception,
      // and convert it to an IOException here.
      throw new IOException(e.getMessage(), e);
    }
  }

  public static FilesetManifest constructFilesetManifest(
      List<FilesetOutputSymlink> outputSymlinks,
      PathFragment targetPrefix,
      RelativeSymlinkBehavior relSymlinkBehavior,
      PathFragment execRoot)
      throws IOException {
    LinkedHashMap<PathFragment, String> entries = new LinkedHashMap<>();
    Map<PathFragment, String> relativeLinks = new HashMap<>();
    Map<String, FileArtifactValue> artifactValues = new HashMap<>();
    for (FilesetOutputSymlink outputSymlink : outputSymlinks) {
      PathFragment fullLocation = targetPrefix.getRelative(outputSymlink.getName());
      PathFragment linkTarget = outputSymlink.reconstituteTargetPath(execRoot);
      String artifact = Strings.emptyToNull(linkTarget.getPathString());
      addSymlinkEntry(artifact, fullLocation, relSymlinkBehavior, entries, relativeLinks);
      if (outputSymlink.getMetadata() instanceof FileArtifactValue) {
        artifactValues.put(artifact, (FileArtifactValue) outputSymlink.getMetadata());
      }
    }
    try {
      return constructFilesetManifest(entries, relativeLinks, artifactValues);
    } catch (IORuntimeException e) {
      throw new IOException(e.getMessage(), e);
    }
  }

  private static final class ManifestLineProcessor implements LineProcessor<FilesetManifest> {
    private final String workspaceName;
    private final PathFragment targetPrefix;
    private final RelativeSymlinkBehavior relSymlinkBehavior;

    private int lineNum;
    private final LinkedHashMap<PathFragment, String> entries = new LinkedHashMap<>();
    // Resolution order of relative links can affect the outcome of the resolution. In particular,
    // if there's a symlink to a symlink, then resolution fails if the first symlink is resolved
    // first, but works if the second symlink is resolved first.
    private final LinkedHashMap<PathFragment, String> relativeLinks = new LinkedHashMap<>();

    ManifestLineProcessor(
        String workspaceName,
        PathFragment targetPrefix,
        RelativeSymlinkBehavior relSymlinkBehavior) {
      this.workspaceName = workspaceName;
      this.targetPrefix = targetPrefix;
      this.relSymlinkBehavior = relSymlinkBehavior;
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

      String artifact;
      PathFragment location;
      int pos = line.indexOf(' ');
      if (pos == -1) {
        location = PathFragment.create(line);
        artifact = null;
      } else {
        location = PathFragment.create(line.substring(0, pos));
        String targetPath = line.substring(pos + 1);
        artifact = targetPath.isEmpty() ? null : targetPath;

        if (!workspaceName.isEmpty()) {
          if (!location.getSegment(0).equals(workspaceName)) {
            throw new IOException(
                String.format(
                    "fileset manifest line must start with '%s': '%s'", workspaceName, location));
          } else {
            // Erase "<workspaceName>/" prefix.
            location = location.subFragment(1);
          }
        }
      }

      PathFragment fullLocation = targetPrefix.getRelative(location);
      addSymlinkEntry(artifact, fullLocation, relSymlinkBehavior, entries, relativeLinks);
      return true;
    }

    @Override
    public FilesetManifest getResult() {
      return constructFilesetManifest(entries, relativeLinks, ImmutableMap.of());
    }
  }

  private static void addSymlinkEntry(
      String artifact,
      PathFragment fullLocation,
      RelativeSymlinkBehavior relSymlinkBehavior,
      LinkedHashMap<PathFragment, String> entries,
      Map<PathFragment, String> relativeLinks)
      throws IOException {
    if (!entries.containsKey(fullLocation)) {
      boolean isRelativeSymlink = artifact != null && !artifact.startsWith("/");
      if (isRelativeSymlink && relSymlinkBehavior.equals(RelativeSymlinkBehavior.ERROR)) {
        throw new IOException(String.format("runfiles target is not absolute: %s", artifact));
      }
      if (!isRelativeSymlink || relSymlinkBehavior.equals(RelativeSymlinkBehavior.RESOLVE)) {
        entries.put(fullLocation, artifact);
        if (artifact != null && !artifact.startsWith("/")) {
          relativeLinks.put(fullLocation, artifact);
        }
      }
    }
  }

  private static final int MAX_SYMLINK_TRAVERSALS = 256;

  private static FilesetManifest constructFilesetManifest(
      Map<PathFragment, String> entries,
      Map<PathFragment, String> relativeLinks,
      Map<String, FileArtifactValue> artifactValues)
      throws IORuntimeException {
    // Resolve relative symlinks. Note that relativeLinks only contains entries in RESOLVE mode.
    // We must find targets for these symlinks that are not inside the Fileset itself.
    for (Map.Entry<PathFragment, String> e : relativeLinks.entrySet()) {
      PathFragment location = e.getKey();
      String value = e.getValue();
      String actual = Preconditions.checkNotNull(value, e);
      Preconditions.checkState(!actual.startsWith("/"), e);
      PathFragment actualLocation = location;
      // Recursively resolve relative symlinks.
      LinkedHashSet<String> seen = new LinkedHashSet<>();
      int i = 0;
      do {
        actualLocation = actualLocation.getParentDirectory().getRelative(actual);
        actual = entries.get(actualLocation);
      } while (actual != null
          && !actual.startsWith("/")
          && seen.add(actual)
          && ++i < MAX_SYMLINK_TRAVERSALS);
      if (actual == null) {
        // We've found a relative symlink that points out of the fileset. We should really always
        // throw here, but current behavior is that we tolerate such symlinks when they occur in
        // runfiles, which is the only time this code is hit.
        // TODO(b/113128395): throw here.
        logger.warning(
            "Symlink "
                + location
                + " (transitively) points to "
                + actualLocation
                + " that is not in this fileset (or was pruned because of a cycle)");
        entries.remove(location);
      } else if (i >= MAX_SYMLINK_TRAVERSALS) {
        logger.warning(
            "Symlink "
                + location
                + " is part of a chain of length at least "
                + i
                + " which exceeds Blaze's maximum allowable symlink chain length");
        entries.remove(location);
      } else if (!actual.startsWith("/")) {
        // TODO(b/113128395): throw here.
        logger.warning("Symlink " + location + " forms a symlink cycle: " + seen);
        // Removing the entry here will lead to slightly vague log lines for the other entries in
        // the cycle, since they will fail when they don't find this entry, as opposed to
        // discovering their own cycles. But this log line should be informative enough.
        entries.remove(location);
      } else {
        entries.put(location, actual);
      }
    }
    return new FilesetManifest(entries, artifactValues);
  }

  private final Map<PathFragment, String> entries;
  private final Map<String, FileArtifactValue> artifactValues;

  private FilesetManifest(Map<PathFragment, String> entries,
      Map<String, FileArtifactValue> artifactValues) {
    this.entries = Collections.unmodifiableMap(entries);
    this.artifactValues = artifactValues;
  }

  public Map<PathFragment, String> getEntries() {
    return entries;
  }

  public Map<String, FileArtifactValue> getArtifactValues() {
    return artifactValues;
  }
}
