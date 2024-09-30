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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Strings;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Representation of a Fileset manifest.
 */
public final class FilesetManifest {
  private static final int MAX_SYMLINK_TRAVERSALS = 256;
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Mode that determines how to handle relative target paths. */
  public enum RelativeSymlinkBehavior {
    /** Ignore any relative target paths. */
    IGNORE,

    /** Give an error if a relative target path is encountered. */
    ERROR,

    /** Resolve all relative target paths. */
    RESOLVE,

    /** Fully resolve all relative paths, even those pointing to internal directories. */
    RESOLVE_FULLY
  }

  /**
   * Shadow of {@link RelativeSymlinkBehavior} without the {@link RelativeSymlinkBehavior#ERROR}
   * value for callers who know there won't be an error thrown when constructing the manifest.
   */
  public enum RelativeSymlinkBehaviorWithoutError {
    /** Ignore any relative target paths. */
    IGNORE(RelativeSymlinkBehavior.IGNORE),

    /** Resolve all relative target paths. */
    RESOLVE(RelativeSymlinkBehavior.RESOLVE),

    /** Fully resolve all relative paths, even those pointing to internal directories. */
    RESOLVE_FULLY(RelativeSymlinkBehavior.RESOLVE_FULLY);

    private final RelativeSymlinkBehavior target;

    RelativeSymlinkBehaviorWithoutError(RelativeSymlinkBehavior target) {
      this.target = target;
    }
  }

  /**
   * Constructs a FilesetManifest from the given {@code outputSymlinks}, processing relative
   * symlinks according to {@code relSymlinkBehavior}. Use when {@link
   * RelativeSymlinkBehavior#ERROR} is guaranteed not to be the behavior.
   */
  public static FilesetManifest constructFilesetManifestWithoutError(
      List<FilesetOutputSymlink> outputSymlinks,
      PathFragment targetPrefix,
      RelativeSymlinkBehaviorWithoutError relSymlinkBehavior) {
    try {
      return constructFilesetManifest(outputSymlinks, targetPrefix, relSymlinkBehavior.target);
    } catch (ForbiddenRelativeSymlinkException e) {
      throw new IllegalStateException(
          "Can't throw forbidden symlink exception unless behavior is ERROR: "
              + relSymlinkBehavior
              + ", "
              + targetPrefix
              + ", "
              + outputSymlinks,
          e);
    }
  }

  public static FilesetManifest constructFilesetManifest(
      List<FilesetOutputSymlink> outputSymlinks,
      PathFragment targetPrefix,
      RelativeSymlinkBehavior relSymlinkBehavior)
      throws ForbiddenRelativeSymlinkException {
    LinkedHashMap<PathFragment, String> entries = new LinkedHashMap<>();
    Map<PathFragment, String> relativeLinks = new HashMap<>();
    Map<String, FileArtifactValue> artifactValues = new HashMap<>();
    for (FilesetOutputSymlink outputSymlink : outputSymlinks) {
      PathFragment fullLocation = targetPrefix.getRelative(outputSymlink.getName());
      String artifact = Strings.emptyToNull(outputSymlink.getTargetPath().getPathString());
      if (isRelativeSymlink(outputSymlink)) {
        addRelativeSymlinkEntry(artifact, fullLocation, relSymlinkBehavior, relativeLinks);
      } else {
        // Symlinks are already deduplicated by name in SkyframeFilesetManifestAction.
        checkState(
            entries.put(fullLocation, artifact) == null,
            "Duplicate fileset entry at %s",
            fullLocation);
      }
      if (outputSymlink.getMetadata() instanceof FileArtifactValue) {
        artifactValues.put(artifact, (FileArtifactValue) outputSymlink.getMetadata());
      }
    }
    resolveRelativeSymlinks(entries, relativeLinks, targetPrefix.isAbsolute(), relSymlinkBehavior);
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
      throws ForbiddenRelativeSymlinkException {
    switch (relSymlinkBehavior) {
      case ERROR -> throw new ForbiddenRelativeSymlinkException(artifact);
      case RESOLVE, RESOLVE_FULLY ->
          checkState(
              relativeLinks.put(fullLocation, artifact) == null,
              "Duplicate fileset entry at %s",
              fullLocation);
      case IGNORE -> {
        // Do nothing.
      }
    }
  }

  /** Fully resolve relative symlinks including internal directory symlinks. */
  private static void fullyResolveRelativeSymlinks(
      Map<PathFragment, String> entries,
      Map<PathFragment, String> relativeLinks,
      boolean absolute) {
    try {
      // Construct an in-memory Filesystem containing all the non-relative-symlink entries in the
      // Fileset. Treat these as regular files in the filesystem whose contents are the "real"
      // symlink
      // pointing out of the Fileset. For relative symlinks, we encode these as symlinks in the
      // in-memory Filesystem. This allows us to then crawl the filesystem for files. Any readable
      // file is a valid part of the FilesetManifest. Dangling internal links or symlink cycles will
      // be discovered by the in-memory filesystem.
      // (Choice of digest function is irrelevant).
      InMemoryFileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
      Path root = fs.getPath("/");
      for (Map.Entry<PathFragment, String> e : entries.entrySet()) {
        PathFragment location = e.getKey();
        Path locationPath = root.getRelative(location);
        locationPath.getParentDirectory().createDirectoryAndParents();
        FileSystemUtils.writeContentAsLatin1(locationPath, Strings.nullToEmpty(e.getValue()));
      }
      for (Map.Entry<PathFragment, String> e : relativeLinks.entrySet()) {
        PathFragment location = e.getKey();
        Path locationPath = fs.getPath("/").getRelative(location);
        PathFragment value = PathFragment.create(checkNotNull(e.getValue(), e));
        checkState(!value.isAbsolute(), e);

        locationPath.getParentDirectory().createDirectoryAndParents();
        locationPath.createSymbolicLink(value);
      }

      addSymlinks(root, entries, absolute);
    } catch (IOException e) {
      throw new IllegalStateException("InMemoryFileSystem can't throw", e);
    }
  }

  private static void addSymlinks(Path root, Map<PathFragment, String> entries, boolean absolute)
      throws IOException {
    for (Path path : root.getDirectoryEntries()) {
      try {
        if (path.isDirectory()) {
          addSymlinks(path, entries, absolute);
        } else {
          String contents = new String(FileSystemUtils.readContentAsLatin1(path));
          entries.put(
              absolute ? path.asFragment() : path.asFragment().toRelative(),
              Strings.emptyToNull(contents));
        }
      } catch (IOException e) {
        logger.atWarning().log("Symlink %s is dangling or cyclic: %s", path, e.getMessage());
      }
    }
  }

  /**
   * Resolves relative symlinks and puts them in the {@code entries} map.
   *
   * <p>Note that {@code relativeLinks} should only contain entries in {@link
   * RelativeSymlinkBehavior#RESOLVE} or {@link RelativeSymlinkBehavior#RESOLVE_FULLY} mode.
   */
  private static void resolveRelativeSymlinks(
      Map<PathFragment, String> entries,
      Map<PathFragment, String> relativeLinks,
      boolean absolute,
      RelativeSymlinkBehavior relSymlinkBehavior) {
    if (relSymlinkBehavior == RelativeSymlinkBehavior.RESOLVE_FULLY && !relativeLinks.isEmpty()) {
      fullyResolveRelativeSymlinks(entries, relativeLinks, absolute);
    } else if (relSymlinkBehavior == RelativeSymlinkBehavior.RESOLVE) {
      for (Map.Entry<PathFragment, String> e : relativeLinks.entrySet()) {
        PathFragment location = e.getKey();
        String value = e.getValue();
        String actual = checkNotNull(value, e);
        checkState(!actual.startsWith("/"), e);
        PathFragment actualLocation = location;

        // Recursively resolve relative symlinks.
        LinkedHashSet<String> seen = new LinkedHashSet<>();
        int traversals = 0;
        do {
          actualLocation = actualLocation.getParentDirectory().getRelative(actual);
          actual = relativeLinks.get(actualLocation);
        } while (++traversals <= MAX_SYMLINK_TRAVERSALS && actual != null && seen.add(actual));

        if (traversals >= MAX_SYMLINK_TRAVERSALS) {
          logger.atWarning().log(
              "Symlink %s is part of a chain of length at least %d"
                  + " which exceeds Blaze's maximum allowable symlink chain length",
              location, traversals);
        } else if (actual != null) {
          // TODO(b/113128395): throw here.
          logger.atWarning().log("Symlink %s forms a symlink cycle: %s", location, seen);
        } else if (!entries.containsKey(actualLocation)) {
          // We've found a relative symlink that points out of the fileset. We should really always
          // throw here, but current behavior is that we tolerate such symlinks when they occur in
          // runfiles, which is the only time this code is hit.
          // TODO(b/113128395): throw here.
          logger.atWarning().log(
              "Symlink %s (transitively) points to %s"
                  + " that is not in this fileset (or was pruned because of a cycle)",
              location, actualLocation);
        } else {
          // We have successfully resolved the symlink.
          entries.put(location, entries.get(actualLocation));
        }
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

  /** Exception indicating that a relative symlink was encountered but not permitted. */
  public static final class ForbiddenRelativeSymlinkException
      extends ForbiddenActionInputException {
    private ForbiddenRelativeSymlinkException(String symlinkTarget) {
      super("Fileset symlink " + symlinkTarget + " is not absolute");
    }
  }
}
