// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.fileset;

import com.google.devtools.build.lib.packages.FilesetEntry;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Map;
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
public interface FilesetLinks {

  /**
   * Get late directory information for a source.
   *
   * @param src The source to search for.
   * @return The late directory info, or null if none was found.
   */
  public LateDirectoryInfo getLateDirectoryInfo(PathFragment src);

  public boolean putLateDirectoryInfo(PathFragment src, LateDirectoryInfo lateDir);

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
      throws IOException;

  /**
   * Add all late directories as symlinks. This function should be called only
   * after all recursions have completed, but before getData or getSymlinks are
   * called.
   */
  public void addLateDirectories() throws IOException;

  /**
   * Adds the given symlink to the tree.
   *
   * @param fromFrag The root-relative symlink path.
   * @param toFrag The symlink target.
   * @return true iff the symlink was added.
   */
  public boolean addLink(PathFragment fromFrag, PathFragment toFrag, String dataVal);

  /**
   * @return The unmodifiable map of symlinks.
   */
  public Map<PathFragment, PathFragment> getSymlinks();

  /**
   * @return The unmodifiable map of metadata.
   */
  public Map<PathFragment, String> getData();

  /**
   * A data structure for containing all the information about a directory that
   * is late-added. This means the directory is skipped unless we need to
   * recurse into it later. If the directory is never recursed into, we will
   * create a symlink directly to it.
   */
  public static final class LateDirectoryInfo {
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
    public Path getTarget() {
      return target;
    }

    /**
     * @return The source path for the symlink. The source is the place the
     *     symlink will be written.  */
    public PathFragment getSrc() {
      return src;
    }

    /**
     * @return Whether we should show a warning if we cross a package boundary
     * when recursing into this directory.
     */
    public SubpackageMode getPkgMode() {
      return pkgMode;
    }

    /**
     * @return The metadata we will write into the manifest if we symlink to
     * this directory.
     */
    public String getMetadata() {
      return metadata;
    }

    /**
     * @return How to perform the symlinking if the source happens to be a
     * symlink itself.
     */
    public FilesetEntry.SymlinkBehavior getTargetSymlinkBehavior() {
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
    public boolean shouldAdd() {
      return !added.getAndSet(true);
    }

    /**
     * Create a stub LateDirectoryInfo that is already marked as added.
     * @return The new LateDirectoryInfo object.
     */
    public static LateDirectoryInfo createStub() {
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
    public static LateDirectoryInfo create(Path target, PathFragment src, SubpackageMode pkgMode,
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

  /** How to handle filesets that cross subpackages. */
  public static enum SubpackageMode {
    ERROR, WARNING, IGNORE;
  }
}
