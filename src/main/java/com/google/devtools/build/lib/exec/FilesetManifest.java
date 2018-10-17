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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import javax.annotation.Nullable;

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

  public static FilesetManifest constructFilesetManifest(
      List<FilesetOutputSymlink> outputSymlinks,
      PathFragment targetPrefix,
      RelativeSymlinkBehavior relSymlinkBehavior)
      throws IOException {
    LinkedHashMap<PathFragment, String> entries = new LinkedHashMap<>();
    Map<PathFragment, String> relativeLinks = new HashMap<>();
    Map<String, FileArtifactValue> artifactValues = new HashMap<>();
    for (FilesetOutputSymlink outputSymlink : outputSymlinks) {
      PathFragment fullLocation = targetPrefix.getRelative(outputSymlink.getName());
      String artifact = Strings.emptyToNull(outputSymlink.getTargetPath().getPathString());
      if (isRelativeSymlink(outputSymlink)) {
        addRelativeSymlinkEntry(artifact, fullLocation, relSymlinkBehavior, relativeLinks);
      } else if (!entries.containsKey(fullLocation)) { // Keep consistent behavior: no overwriting.
        entries.put(fullLocation, artifact);
      }
      if (outputSymlink.getMetadata() instanceof FileArtifactValue) {
        artifactValues.put(artifact, (FileArtifactValue) outputSymlink.getMetadata());
      }
    }
    resolveRelativeSymlinks(entries, relativeLinks);
    return new FilesetManifest(entries, artifactValues);
  }

  private static boolean isRelativeSymlink(FilesetOutputSymlink symlink) {
    return !symlink.getTargetPath().isEmpty()
        && !symlink.getTargetPath().isAbsolute()
        && !symlink.isRelativeToExecRoot();
  }

  /** Potentially adds the relative symlink to the map, depending on {@code relSymlinkBehavior}. */
  private static void addRelativeSymlinkEntry(
      @Nullable String artifact,
      PathFragment fullLocation,
      RelativeSymlinkBehavior relSymlinkBehavior,
      Map<PathFragment, String> relativeLinks)
      throws IOException {
    switch (relSymlinkBehavior) {
      case ERROR:
        throw new IOException("runfiles target is not absolute: " + artifact);
      case RESOLVE:
        if (!relativeLinks.containsKey(fullLocation)) { // Keep consistent behavior: no overwriting.
          relativeLinks.put(fullLocation, artifact);
        }
        break;
      case IGNORE:
        break; // Do nothing.
    }
  }

  private static final int MAX_SYMLINK_TRAVERSALS = 256;

  /**
   * Resolves relative symlinks and puts them in the {@code entries} map.
   *
   * <p>Note that {@code relativeLinks} should only contain entries in {@link
   * RelativeSymlinkBehavior#RESOLVE} mode.
   */
  private static void resolveRelativeSymlinks(
      Map<PathFragment, String> entries, Map<PathFragment, String> relativeLinks) {
    for (Map.Entry<PathFragment, String> e : relativeLinks.entrySet()) {
      PathFragment location = e.getKey();
      String value = e.getValue();
      String actual = Preconditions.checkNotNull(value, e);
      Preconditions.checkState(!actual.startsWith("/"), e);
      PathFragment actualLocation = location;

      // Recursively resolve relative symlinks.
      LinkedHashSet<String> seen = new LinkedHashSet<>();
      int traversals = 0;
      do {
        actualLocation = actualLocation.getParentDirectory().getRelative(actual);
        actual = relativeLinks.get(actualLocation);
      } while (++traversals <= MAX_SYMLINK_TRAVERSALS && actual != null && seen.add(actual));

      if (traversals >= MAX_SYMLINK_TRAVERSALS) {
        logger.warning(
            "Symlink "
                + location
                + " is part of a chain of length at least "
                + traversals
                + " which exceeds Blaze's maximum allowable symlink chain length");
      } else if (actual != null) {
        // TODO(b/113128395): throw here.
        logger.warning("Symlink " + location + " forms a symlink cycle: " + seen);
      } else if (!entries.containsKey(actualLocation)) {
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
      } else {
        // We have successfully resolved the symlink.
        entries.put(location, entries.get(actualLocation));
      }
    }
  }

  private final Map<PathFragment, String> entries;
  private final Map<String, FileArtifactValue> artifactValues;

  private FilesetManifest(
      Map<PathFragment, String> entries, Map<String, FileArtifactValue> artifactValues) {
    this.entries = Collections.unmodifiableMap(entries);
    this.artifactValues = artifactValues;
  }

  /**
   * Returns a mapping of symlink name to its target path.
   *
   * <p>Values in this map can be:
   *
   * <ul>
   *   <li>An absolute path.
   *   <li>A relative path, which should be considered relative to the exec root.
   *   <li>{@code null}, which represents an empty file.
   * </ul>
   */
  public Map<PathFragment, String> getEntries() {
    return entries;
  }

  /**
   * Returns a mapping of target path to {@link FileArtifactValue}.
   *
   * <p>The keyset of this map is a subset of the values in the map returned by {@link #getEntries}.
   */
  public Map<String, FileArtifactValue> getArtifactValues() {
    return artifactValues;
  }
}
