// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Canonicalizes paths like {@link FileSystem#resolveSymbolicLinks}, while storing the intermediate
 * results in a trie so they can be reused by future canonicalizations.
 *
 * <p>This is an implementation detail of {@link RemoteActionFileSystem}, factored out for testing.
 * Because {@link RemoteActionFileSystem} implements a union filesystem and must account for the
 * possibility of symlinks straddling the underlying filesystems, the performance of large
 * filesystem scans can be greatly improved with a custom {@link FileSystem#resolveSymbolicLinks}
 * implementation that leverages the trie to avoid repeated work.
 *
 * <p>On case-insensitive filesystems, accessing the same path through different case variations
 * will produce distinct trie entries. This could be fixed, but it's a performance rather than a
 * correctness concern, and shouldn't matter most of the time.
 *
 * <p>Thread-safe: concurrent calls to {@link #resolveSymbolicLinks} are supported. As with {@link
 * FileSystem#resolveSymbolicLinks}, the result is undefined if the filesystem is mutated
 * concurrently.
 */
final class PathCanonicalizer {

  interface Resolver {
    /**
     * Returns the result of {@link FileSystem#readSymbolicLink} if the path is a symlink, otherwise
     * null. All but the last path segment must be canonical.
     *
     * @throws IOException if the file type or symlink target path could not be determined
     */
    @Nullable
    PathFragment resolveOneLink(PathFragment path) throws IOException;
  }

  /** A trie node. */
  private sealed interface Node {}

  /** A trie node corresponding to a symlink. */
  private record SymlinkNode(PathFragment targetPath) implements Node {}

  /** A trie node not corresponding to a symlink. */
  private static final class NonSymlinkNode extends ConcurrentHashMap<String, Node>
      implements Node {
    NonSymlinkNode() {
      super(/* initialCapacity= */ 1);
    }
  }

  private final Resolver resolver;
  private final NonSymlinkNode root = new NonSymlinkNode();

  PathCanonicalizer(Resolver resolver) {
    this.resolver = resolver;
  }

  /** Returns the root node for an absolute path. */
  private NonSymlinkNode getRootNode(PathFragment path) {
    checkArgument(path.isAbsolute());
    // Unix has a single root. Windows has one root per drive.
    if (path.getDriveStrLength() > 1) {
      return (NonSymlinkNode)
          root.computeIfAbsent(path.getDriveStr(), unused -> new NonSymlinkNode());
    }
    return root;
  }

  /**
   * Canonicalizes a path, reusing cached information if possible.
   *
   * @param path the path to canonicalize.
   * @param maxLinks the maximum number of symlinks that can be followed in the process of
   *     canonicalizing the path.
   * @param allowMissingFinalComponent whether the final component may be missing, as when
   *     canonicalizing an input's parent path.
   * @throws FileSymlinkLoopException if too many symlinks had to be followed.
   * @throws IOException if an I/O error occurs
   * @return the canonical path.
   */
  private PathFragment resolveSymbolicLinks(
      PathFragment path, int maxLinks, boolean allowMissingFinalComponent) throws IOException {
    // This code is carefully written to be as fast as possible when the path is already canonical
    // and has been previously cached. Avoid making changes without benchmarking. A tree artifact
    // with hundreds of thousands of files makes for a good benchmark.

    NonSymlinkNode node = getRootNode(path);
    Iterable<String> segments = path.segments();
    int segmentIndex = 0;

    // Loop invariants:
    // - `segmentIndex` is the index of the current `segment` relative to the start of `path`. The
    //   first segment has index 0.
    // - `path` is the absolute path to canonicalize. If `segmentIndex` > 0, `path` is already
    //    canonical up to and including `segmentIndex` - 1.
    // - `node` is the trie node corresponding to the `path` prefix ending with `segmentIndex` - 1,
    //   or to the root path when `segmentIndex` is 0.
    for (String segment : segments) {
      Node nextNode = node.get(segment);
      if (nextNode == null) {
        PathFragment naivePath = path.subFragment(0, segmentIndex + 1);
        PathFragment targetPath;
        try {
          targetPath = resolver.resolveOneLink(naivePath);
        } catch (FileNotFoundException e) {
          if (segmentIndex + 1 == path.segmentCount() && !allowMissingFinalComponent) {
            throw e;
          }
          // Input metadata can exist without its parent directories. Continue in a detached trie
          // so this missing prefix and its descendants are rechecked on the next resolution.
          node = new NonSymlinkNode();
          segmentIndex++;
          continue;
        }
        nextNode =
            node.computeIfAbsent(
                segment,
                unused -> targetPath != null ? new SymlinkNode(targetPath) : new NonSymlinkNode());
      }

      switch (nextNode) {
        case SymlinkNode(PathFragment targetPath) -> {
          if (maxLinks == 0) {
            throw new FileSymlinkLoopException(
                path.getPathString() + FileSystem.ERR_TOO_MANY_SYMLINKS);
          }
          maxLinks--;

          // Compute the path obtained by resolving the symlink.
          // Note that path normalization already handles uplevel references.
          PathFragment newPath;
          if (targetPath.isAbsolute()) {
            newPath = targetPath.getRelative(path.subFragment(segmentIndex + 1));
          } else {
            newPath =
                path.subFragment(0, segmentIndex)
                    .getRelative(targetPath)
                    .getRelative(path.subFragment(segmentIndex + 1));
          }

          // For absolute symlinks, we must start over.
          // For relative symlinks, it would have been possible to restart after the already
          // canonicalized prefix, but they're too rare to be worth optimizing for.
          return resolveSymbolicLinks(newPath, maxLinks, allowMissingFinalComponent);
        }
        case NonSymlinkNode nonSymlinkNode -> {
          node = nonSymlinkNode;
          segmentIndex++;
        }
      }
    }

    return path;
  }

  /**
   * Canonicalizes a path, reusing cached information if possible.
   *
   * <p>Like {@link FileSystem#resolveSymbolicLinks}, except that missing intermediate directories
   * are allowed: input metadata may describe a file without describing its parents. The final
   * component must exist.
   *
   * @param path the path to canonicalize.
   * @throws FileSymlinkLoopException if too many symlinks had to be followed.
   * @throws IOException if an I/O error occurs
   * @return the canonical path.
   */
  PathFragment resolveSymbolicLinks(PathFragment path) throws IOException {
    return resolveSymbolicLinks(
        path, FileSystem.MAX_SYMLINKS, /* allowMissingFinalComponent= */ false);
  }

  /**
   * Canonicalizes only the parent path, allowing missing parent directories. The caller must look
   * up or operate on the final path to determine whether it exists.
   */
  PathFragment resolveSymbolicLinksForParent(PathFragment path) throws IOException {
    checkArgument(path.isAbsolute());
    PathFragment parent = path.getParentDirectory();
    return parent == null
        ? path
        : resolveSymbolicLinks(
                parent, FileSystem.MAX_SYMLINKS, /* allowMissingFinalComponent= */ true)
            .getChild(path.getBaseName());
  }

  /** Removes cached information for a path prefix. */
  void clearPrefix(PathFragment pathPrefix) {
    Node node = getRootNode(pathPrefix);
    NonSymlinkNode parent = null;
    String parentSegment = null;
    Iterator<String> segments = pathPrefix.segments().iterator();
    boolean hasNext = segments.hasNext();

    while (node != null && hasNext) {
      String segment = segments.next();
      hasNext = segments.hasNext();

      switch (node) {
        case SymlinkNode symlinkNode -> {
          // Invalidate all intermediate symlinks.
          if (parent != null) {
            parent.remove(parentSegment);
          }
          return;
        }
        case NonSymlinkNode nonSymlinkNode -> {
          if (!hasNext) {
            // Found the path prefix.
            nonSymlinkNode.remove(segment);
          } else {
            parent = nonSymlinkNode;
            parentSegment = segment;
            node = nonSymlinkNode.get(segment);
          }
        }
      }
    }
  }
}
