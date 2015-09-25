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
package com.google.devtools.build.lib.vfs;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;

import java.io.IOException;

/**
 * A file system that's capable of identifying paths residing outside its scope
 * and using a delegator (such as {@link UnionFileSystem}) to re-route them
 * to appropriate alternative file systems.
 *
 * <p>This is most useful for symlinks, which may ostensibly fall beneath some
 * file system but resolve to paths outside that file system.
 *
 * <p>Note that we don't protect against cross-filesystem circular references.
 * Therefore, care should be taken not to mix two scopable file systems that
 * can reference each other. This theoretical safety cost is balanced by
 * decreased code complexity requirements in implementations.
 */
public abstract class ScopeEscapableFileSystem extends FileSystem {

  private FileSystem delegator;
  protected final PathFragment scopeRoot;
  private boolean enableScopeChecking = true; // Used for testing.

  /**
   * Instantiates a new ScopeEscapableFileSystem.
   *
   * @param scopeRoot the root path for the file system's scope. Any path
   *        that isn't beneath this one is considered out of scope according
   *        to {@link #inScope}. If null, scope checking is disabled. Note
   *        this is not the same thing as {@link FileSystem#rootPath}, which
   *        generally resolves to "/".
   */
  protected ScopeEscapableFileSystem(PathFragment scopeRoot) {
    this.scopeRoot = scopeRoot;
  }

  @VisibleForTesting
  void enableScopeChecking(boolean enable) {
    this.enableScopeChecking = enable;
  }

  /**
   * Sets the delegator used to resolve paths that fall outside this file
   * system's scope.
   *
   * <p>This method is not thread safe. It's intended to be called during
   * instance initialization, not during active usage. The only reason this
   * isn't set as immutable state within the constructor is that the delegator
   * may need a reference to this instance for its own constructor.
   */
  @ThreadHostile
  public void setDelegator(FileSystem delegator) {
    this.delegator = delegator;
  }

  /**
   * Uses the delegator to convert a path fragment to a path that's bound
   * to the file system that manages that path.
   */
  protected Path getDelegatedPath(PathFragment path) {
    Preconditions.checkState(delegator != null);
    return delegator.getPath(path);
  }

  /**
   * Proxy for {@link FileSystem#resolveOneLink} that sends the input path
   * through the delegator.
   */
  protected PathFragment resolveOneLinkWithDelegator(final PathFragment path) throws IOException {
    Preconditions.checkState(delegator != null);
    return delegator.resolveOneLink(getDelegatedPath(path));
  }

  /**
   * Proxy for {@link FileSystem#stat} that sends the input path through
   * the delegator.
   */
  protected FileStatus statWithDelegator(final PathFragment path, final boolean followSymlinks)
      throws IOException {
    Preconditions.checkState(delegator != null);
    return delegator.stat(getDelegatedPath(path), followSymlinks);
  }

  /**
   * Returns true if the given path is within this file system's scope, false
   * otherwise.
   *
   * @param parentDepth the number of segments in the path's parent directory
   *        (only meaningful for paths that begin with ".."). The parent directory
   *        itself is assumed to be in scope.
   * @param normalizedPath input path, expected to be normalized such that all
   *        ".." and "." segments are removed (with the exception of a possible
   *        prefix sequence of contiguous ".." segments)
   */
  protected boolean inScope(int parentDepth, PathFragment normalizedPath) {
    if (scopeRoot == null || !enableScopeChecking) {
      return true;
    } else if (normalizedPath.isAbsolute()) {
      return normalizedPath.startsWith(scopeRoot);
    } else {
      // Efficiency note: we're not accounting for "/scope/root/../root" paths here, i.e. paths
      // that appear to go out of scope but ultimately stay within scope. This may result in
      // unnecessary re-delegation back into the same FS. we're choosing to forgo that
      // optimization under the assumption that such scenarios are rare and unimportant to
      // overall performance. We can always enhance this if needed.
      return parentDepth - leadingParentReferences(normalizedPath) >= scopeRoot.segmentCount();
    }
  }

  /**
   * Given a path that's normalized (no ".." or "." segments), except for a possible
   * prefix sequence of contiguous ".." segments, returns the size of that prefix
   * sequence.
   *
   * <p>Example allowed inputs: "/absolute/path", "relative/path", "../../relative/path".
   * Example disallowed inputs: "/absolute/path/../path2", "relative/../path", "../relative/../p".
   */
  protected int leadingParentReferences(PathFragment normalizedPath) {
    int leadingParentReferences = 0;
    for (int i = 0; i < normalizedPath.segmentCount() &&
        normalizedPath.getSegment(i).equals(".."); i++) {
      leadingParentReferences++;
    }
    return leadingParentReferences;
  }
}
