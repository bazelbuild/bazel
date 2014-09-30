// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.fileset;

import com.google.common.base.Preconditions;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.view.fileset.FilesetUtil.SubpackageMode;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * FilesetLinks manages the set of links added to a Fileset. If two links conflict, the first wins.
 *
 * <p>FilesetLinks is FileSystem-aware.  For example, if you first create a link
 * (a/b/c, foo), a subsequent call to link (a/b, bar) is a no-op.
 * This is because the first link requires us to create a directory "a/b",
 * so "a/b" cannot also link to "bar".
 *
 * <p>TODO(bazel-team): Consider warning if we have such a conflict; we don't do that currently.
 */
final class FilesetLinks {

  /** This is the root of our virtual file system **/
  private final Path rootNode = new InMemoryFileSystem(BlazeClock.instance())
      .getRootDirectory();

  /**
   * A map of all the symlinks in our file system.
   *
   * <p>The links are kept in lexicographic ordering via the TreeMap.
   *
   * <p>While the map operations grow to O(log(n)) from O(1), this
   * is a worthwhile tradeoff (CPU time for I/O time).
   * The extra CPU time is more than made up for in the build-runfiles script,
   * which executes much faster with the ordering.
   * This is likely because directory updates are written at about the
   * same time and are likely to be written as a single batch operation
   * to disk.
   */
  private final Map<PathFragment, PathFragment> links = Maps.newTreeMap();

  /** A map from source of link to its metadata, used for dependency checking.  */
  private final Map<PathFragment, String> data = new HashMap<>();

  /**
   * A map of source directories to late directory information required to
   * scan the directories later, create symlinks to them if we don't need to
   * recurse into them, or decide whether we need to recurse into other
   * directories as we find them.
   */
  private final ConcurrentMap<PathFragment, LateDirectoryInfo> lateDirs = new MapMaker().makeMap();

  /**
   * Get late directory information for a source.
   *
   * @param src The source to search for.
   * @return The late directory info, or null if none was found.
   */
  public LateDirectoryInfo getLateDirectoryInfo(PathFragment src) {
    return lateDirs.get(src);
  }

  /**
   * Set the late directory information for a source. This function does not
   * overwrite existing values.
   *
   * @param src The source to set.
   * @param lateDir The late directory information.
   * @return true iff the source was added.
   */
  public boolean putLateDirectoryInfo(PathFragment src, LateDirectoryInfo lateDir) {
    return lateDirs.putIfAbsent(src, lateDir) == null;
  }

  /**
   * Add specified file as a symlink.
   *
   * The behavior when the target file is a symlink depends on the
   * symlinkBehavior parameter (see comments for FilesetEntry.SymlinkBehavior).
   *
   * @param src The root-relative symlink path.
   * @param target The symlink target.
   */
  public void addFile(PathFragment src, Path target, String metadata,
      FilesetEntry.SymlinkBehavior symlinkBehavior)
      throws IOException {
    // This method accepts a Path from a different FileSystem than the one
    // used inside this class. Care is taken here to separate the Paths
    // referring the two different file systems.

    if (!target.isSymbolicLink()) {
      addLink(src, target.asFragment(), metadata);
      return;
    }

    switch (symlinkBehavior) {
      case COPY:
        addLink(src, target.readSymbolicLink(), metadata);
        break;

      case DEREFERENCE:
        try {
          PathFragment finalTarget = target.resolveSymbolicLinks().asFragment();
          addLink(src, finalTarget, metadata);
        } catch (FileNotFoundException e) {
          throw new DanglingSymlinkException(target.toString());
        }
        break;

      default:
        throw new IllegalStateException("Unhandled FilesetEntry.SymlinkBehavior: " +
            symlinkBehavior);
    }
  }

  /**
   * Add all late directories as symlinks. This function should be called only
   * after all recursals have completed, but before getData or getSymlinks are
   * called.
   */
  public synchronized void addLateDirectories() throws IOException {
    // lateDirs can contain the root directory, but only as a stub, which is,
    // by definition, already marked as added, so reading the symlink is safe.
    for (LateDirectoryInfo lateDir : lateDirs.values()) {
      if (lateDir.shouldAdd()) {
        addFile(lateDir.getSrc(), lateDir.getTarget(), lateDir.getMetadata(),
            lateDir.getTargetSymlinkBehavior());
      }
    }
  }

  /**
   * Adds the given symlink to the tree.
   *
   * @param fromFrag The root-relative symlink path.
   * @param toFrag The symlink target.
   * @return true iff the symlink was added.
   */
  public synchronized boolean addLink(PathFragment fromFrag, PathFragment toFrag, String dataVal) {
    try {
      Path from = rootNode.getRelative(fromFrag);
      Path fromDir = from.getParentDirectory();

      // First, make sure we can actually create the symlink.
      if (from.exists(Symlinks.NOFOLLOW) || !canMakeDirectory(fromDir)) {
        return false;
      }

      FileSystemUtils.createDirectoryAndParents(fromDir);
      if (fromDir.isDirectory(Symlinks.NOFOLLOW) && !from.exists(Symlinks.NOFOLLOW)) {
        from.createSymbolicLink(toFrag);
        links.put(fromFrag, toFrag);
        data.put(fromFrag, dataVal);
        return true;
      } else {
        // We already verified that we could create the symlink,
        // so we shouldn't get here.
        throw new IllegalStateException("Unexpected failure creating symlink " + fromFrag +
                                        " to " + toFrag);
      }
    } catch (IOException e) {
      // This is unexpected: The FileSystem is in memory and the only calls to it
      // are from this method (and should always succeed without throwing).
      throw new IllegalStateException(e);
    }
  }

  /**
   * Return true iff dir is, or can be created, as a real (non-symlink)
   * directory.
   *
   * @param dir The directory.
   * @return true iff dir can be made into or already is a directory.
   */
  private static boolean canMakeDirectory(Path dir) {
    // TODO(bazel-team): What's the point of this logic?  Why not just optimistically attempt the
    // operation and handle failure, instead of computing a complex predicate to predict whether
    // it will succeed?
    while (true) {
      try {
        // If it's a directory, we should be able to create
        // the child directories and symlinks.
        return dir.stat(Symlinks.NOFOLLOW).isDirectory();
      } catch (IOException e) {
        dir = dir.getParentDirectory();
      }
    }
  }

  /**
   * @return The unmodifiable map of symlinks.
   */
  public synchronized Map<PathFragment, PathFragment> getSymlinks() {
    return Collections.unmodifiableMap(links);
  }


  /**
   * @return The unmodifiable map of metadata.
   */
  public synchronized Map<PathFragment, String> getData() {
    return Collections.unmodifiableMap(data);
  }

  /**
   * A data structure for containing all the information about a directory that
   * is late-added. This means the directory is skipped unless we need to
   * recurse into it later. If the directory is never recursed into, we will
   * create a symlink directly to it.
   */
  static final class LateDirectoryInfo {
    // The constructors are private. Use the factory functions below to create
    // instances of this class.

    /** Construct a stub LateDirectoryInfo object. */
    private LateDirectoryInfo() {
      this.added = new AtomicBoolean(true);

      // Shut up the compiler.
      this.target = null;
      this.src = null;
      this.pkgMode = SubpackageMode.IGNORE;
      this.metadata = null;
      this.symlinkBehavior = null;
    }

    /** Construct a normal LateDirectoryInfo object. */
    private LateDirectoryInfo(Path target, PathFragment src, SubpackageMode pkgMode,
        String metadata, FilesetEntry.SymlinkBehavior symlinkBehavior) {
      this.target = target;
      this.src = src;
      this.pkgMode = pkgMode;
      this.metadata = metadata;
      this.symlinkBehavior = symlinkBehavior;
      this.added = new AtomicBoolean(false);
    }

    /** @return The target path for the symlink. The target is the referent. */
    Path getTarget() {
      return target;
    }

    /**
     * @return The source path for the symlink. The source is the place the
     *     symlink will be written.  */
    PathFragment getSrc() {
      return src;
    }

    /**
     * @return Whether we should show a warning if we cross a package boundary
     * when recursing into this directory.
     */
    SubpackageMode getPkgMode() {
      return pkgMode;
    }

    /**
     * @return The metadata we will write into the manifest if we symlink to
     * this directory.
     */
    String getMetadata() {
      return metadata;
    }

    /**
     * @return How to perform the symlinking if the source happens to be a
     * symlink itself.
     */
    FilesetEntry.SymlinkBehavior getTargetSymlinkBehavior() {
      return Preconditions.checkNotNull(symlinkBehavior,
          "should not call this method on stub instances");
    }

    /**
     * Atomically checks if the late directory has been added to the manifest
     * and marks it as added. If this function returns true, it is the
     * responsibility of the caller to recurse into the late directory.
     * Otherwise, some other caller has already, or is in the process of
     * recursing into it.
     * @return Whether the caller should recurse into the late directory.
     */
    boolean shouldAdd() {
      return !added.getAndSet(true);
    }

    /**
     * Create a stub LateDirectoryInfo that is already marked as added.
     * @return The new LateDirectoryInfo object.
     */
    static LateDirectoryInfo createStub() {
      return new LateDirectoryInfo();
    }

    /**
     * Create a LateDirectoryInfo object with the specified attributes.
     * @param target The directory to which the symlinks will refer.
     * @param src    The location at which to create the symlink.
     * @param pkgMode How to handle recursion into another package.
     * @param metadata The metadata for the directory to write into the
     *     manifest if we symlink it directly.
     * @return The new LateDirectoryInfo object.
     */
    static LateDirectoryInfo create(Path target, PathFragment src, SubpackageMode pkgMode,
        String metadata, FilesetEntry.SymlinkBehavior symlinkBehavior) {
      return new LateDirectoryInfo(target, src, pkgMode, metadata, symlinkBehavior);
    }

    /**
     * The target directory to which the symlink will point.
     * Note this is a real path on the filesystem and can't be compared to src
     * or any source (key) in the links map.
     */
    private final Path target;

    /** The referent of the symlink. */
    private final PathFragment src;

    /** Whether to show cross package boundary warnings / errors.  */
    private final SubpackageMode pkgMode;

    /** The metadata to write into the manifest file.  */
    private final String metadata;

    /** How to perform the symlinking if the source happens to be a symlink itself. */
    private final FilesetEntry.SymlinkBehavior symlinkBehavior;

    /** Whether the directory has already been recursed into.  */
    private final AtomicBoolean added;
  }
}
