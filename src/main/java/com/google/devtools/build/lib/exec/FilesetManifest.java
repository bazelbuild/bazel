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

import com.google.common.io.LineProcessor;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Representation of a Fileset manifest.
 */
public final class FilesetManifest {
  /**
   * Mode that determines how to handle relative target paths.
   */
  public enum RelativeSymlinkBehavior {
    /** Ignore any relative target paths. */
    IGNORE,

    /** Give an error if a relative target path is encountered. */
    ERROR,

    /**
     * Attempt to locally resolve the relative target path. Consider a manifest with two entries,
     * foo points to bar, and bar points to the absolute path /foobar. In that case, we can
     * determine that foo actually points at /foobar. Throw an exception if the local resolution
     * fails, e.g., if the target is not in the current manifest, or if it points at another
     * symlink (we could theoretically resolve recursively, but that's more complexity).
     */
    RESOLVE;
  }

  public static FilesetManifest parseManifestFile(
      Artifact manifest,
      Path execRoot,
      String workspaceName,
      RelativeSymlinkBehavior relSymlinkBehavior)
          throws IOException {
    return parseManifestFile(
        manifest.getExecPath(),
        execRoot,
        workspaceName,
        relSymlinkBehavior);
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
    } catch (IllegalStateException e) {
      // We can't throw IOException from getResult below, so we instead use an unchecked exception,
      // and convert it to an IOException here.
      throw new IOException(e.getMessage(), e);
    }
  }

  private static final class ManifestLineProcessor implements LineProcessor<FilesetManifest> {
    private final String workspaceName;
    private final PathFragment targetPrefix;
    private final RelativeSymlinkBehavior relSymlinkBehavior;

    private int lineNum;
    private final Map<PathFragment, String> entries = new LinkedHashMap<>();
    private final Map<PathFragment, String> relativeLinks = new HashMap<>();

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
            location = location.subFragment(1, location.segmentCount());
          }
        }
      }

      PathFragment fullLocation = targetPrefix.getRelative(location);
      if (!entries.containsKey(fullLocation)) {
        boolean isRelativeSymlink = artifact != null && !artifact.startsWith("/");
        if (isRelativeSymlink && relSymlinkBehavior.equals(RelativeSymlinkBehavior.ERROR)) {
          throw new IOException(String.format("runfiles target is not absolute: %s", artifact));
        }
        if (!isRelativeSymlink
            || relSymlinkBehavior.equals(RelativeSymlinkBehavior.RESOLVE)) {
          entries.put(fullLocation, artifact);
          if (artifact != null && !artifact.startsWith("/")) {
            relativeLinks.put(fullLocation, artifact);
          }
        }
      }
      return true;
    }

    @Override
    public FilesetManifest getResult() {
      // Resolve relative symlinks if possible. Note that relativeLinks only contains entries in
      // RESOLVE mode.
      for (Map.Entry<PathFragment, String> e : relativeLinks.entrySet()) {
        PathFragment location = e.getKey();
        String value = e.getValue();
        PathFragment actualLocation = location.getParentDirectory().getRelative(value);
        String actual = entries.get(actualLocation);
        boolean isActualAcceptable = actual == null || actual.startsWith("/");
        if (!entries.containsKey(actualLocation) || !isActualAcceptable) {
          throw new IllegalStateException(
              String.format(
                  "runfiles target '%s' is not absolute, and could not be resolved in the same "
                  + "Fileset", value));
        }
        entries.put(location, actual);
      }
      return new FilesetManifest(entries);
    }
  }

  private final Map<PathFragment, String> entries;

  private FilesetManifest(Map<PathFragment, String> entries) {
    this.entries = Collections.unmodifiableMap(entries);
  }

  public Map<PathFragment, String> getEntries() {
    return entries;
  }
}
