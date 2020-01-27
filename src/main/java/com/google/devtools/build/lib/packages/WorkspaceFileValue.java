// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.StarlarkThread.Extension;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Objects;

/**
 * A SkyValue that contains the result of the parsing of one part of the WORKSPACE file. The parsing
 * of the WORKSPACE file is split before each series of load statement because we need to resolve
 * repositories before being able to load from those repositories.
 */
public class WorkspaceFileValue implements SkyValue {
  public static final SkyFunctionName WORKSPACE_FILE =
      SkyFunctionName.createHermetic("WORKSPACE_FILE");

  /** Argument for the SkyKey to request a WorkspaceFileValue. */
  @Immutable
  @AutoCodec
  public static class WorkspaceFileKey implements SkyKey {
    private static final Interner<WorkspaceFileKey> interner = BlazeInterners.newWeakInterner();

    private final RootedPath path;
    private final int idx;

    private WorkspaceFileKey(RootedPath path, int idx) {
      this.path = path;
      this.idx = idx;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static WorkspaceFileKey create(RootedPath path, int idx) {
      return interner.intern(new WorkspaceFileKey(path, idx));
    }

    public RootedPath getPath() {
      return path;
    }

    public int getIndex() {
      return idx;
    }

    @Override
    public SkyFunctionName functionName() {
      return WORKSPACE_FILE;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof WorkspaceFileKey)) {
        return false;
      }
      WorkspaceFileKey other = (WorkspaceFileKey) obj;
      return Objects.equals(path, other.path) && idx == other.idx;
    }

    @Override
    public int hashCode() {
      return Objects.hash(path.hashCode(), idx);
    }

    @Override
    public String toString() {
      return path + ", " + idx;
    }
  }

  private final Package pkg;
  private final int idx;
  private final RootedPath path;
  private final boolean hasNext;
  private final ImmutableMap<String, Object> bindings;
  private final ImmutableMap<String, Extension> importMap;
  private final ImmutableMap<String, Integer> importToChunkMap;
  private final ImmutableMap<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>>
      repositoryMapping;
  // Mapping of the relative paths of the incrementally updated managed directories
  // to the managing external repositories
  private final ImmutableMap<PathFragment, RepositoryName> managedDirectories;
  // Directories to be excluded from symlinking to the execroot.
  private final ImmutableSortedSet<String> doNotSymlinkInExecrootPaths;

  /**
   * Create a WorkspaceFileValue containing the various values necessary to compute the split
   * WORKSPACE file.
   *
   * @param pkg Package built by agreggating all parts of the split WORKSPACE file up to this one.
   * @param importMap List of imports (i.e., load statements) present in all parts of the split
   *     WORKSPACE file up to this one.
   * @param importToChunkMap Map of all load statements encountered so far to the chunk they
   *     initially appeared in.
   * @param bindings List of top-level variable bindings from the all parts of the split WORKSPACE
   *     file up to this one. The key is the name of the bindings and the value is the actual
   *     object.
   * @param path The rooted path to workspace file to parse.
   * @param idx The index of this part of the split WORKSPACE file (0 for the first one, 1 for the
   *     second one and so on).
   * @param hasNext Is there a next part in the WORKSPACE file or this part the last one?
   * @param managedDirectories Mapping of the relative paths of the incrementally updated managed
   * @param doNotSymlinkInExecrootPaths directories to be excluded from symlinking to the execroot
   */
  public WorkspaceFileValue(
      Package pkg,
      Map<String, Extension> importMap,
      Map<String, Integer> importToChunkMap,
      Map<String, Object> bindings,
      RootedPath path,
      int idx,
      boolean hasNext,
      ImmutableMap<PathFragment, RepositoryName> managedDirectories,
      ImmutableSortedSet<String> doNotSymlinkInExecrootPaths) {
    this.pkg = Preconditions.checkNotNull(pkg);
    this.idx = idx;
    this.path = path;
    this.hasNext = hasNext;
    this.bindings = ImmutableMap.copyOf(bindings);
    this.importMap = ImmutableMap.copyOf(importMap);
    this.importToChunkMap = ImmutableMap.copyOf(importToChunkMap);
    this.repositoryMapping = pkg.getExternalPackageRepositoryMappings();
    this.managedDirectories = managedDirectories;
    this.doNotSymlinkInExecrootPaths = doNotSymlinkInExecrootPaths;
  }

  /**
   * Returns the package. This package may contain errors, in which case the caller should throw
   * a {@link BuildFileContainsErrorsException}.
   */
  public Package getPackage() {
    return pkg;
  }

  @Override
  public String toString() {
    return "<WorkspaceFileValue path=" + path + " idx=" + idx + ">";
  }

  /**
   * Creates a Key for the WorkspaceFileFunction. The path to the workspace file is specified by
   * {@code path}. This key will ask WorkspaceFileFunction to get the {@code idx+1}-th part of the
   * workspace file (so idx = 0 represents the first part, idx = 1, the second part, etc...).
   */
  public static WorkspaceFileKey key(RootedPath path, int idx) {
    return WorkspaceFileKey.create(path, idx);
  }

  public static WorkspaceFileKey key(RootedPath path) {
    return key(path, 0);
  }

  /**
   * Get the key for the next WorkspaceFileValue or null if this value is the last part of the
   * workspace file.
   */
  public SkyKey next() {
    if (hasNext) {
      return key(path, idx + 1);
    } else {
      return null;
    }
  }

  /**
   * The workspace file parsing is cut in several parts and this function returns the index of the
   * part of the workspace file that this value holds. For the first part, this index will be 0, for
   * the second part, it will be 1 and so on.
   */
  public int getIndex() {
    return idx;
  }

  /**
   * The workspace file parsing is cut in several parts and this function returns true if there is
   * a part following the part holds by this value (or false if this is the last part of the
   * WORKSPACE file.
   *
   * <p>This method is public for serialization of the WorkspaceFileValue, #next() should be used
   * to iterate instead of this method.
   */
  public boolean hasNext() {
    return hasNext;
  }

  public RootedPath getPath() {
    return path;
  }

  public ImmutableMap<String, Object> getBindings() {
    return bindings;
  }

  public ImmutableMap<String, Extension> getImportMap() {
    return importMap;
  }

  public ImmutableMap<String, Integer> getImportToChunkMap() {
    return importToChunkMap;
  }

  public ImmutableMap<RepositoryName, ImmutableMap<RepositoryName, RepositoryName>>
      getRepositoryMapping() {
    return repositoryMapping;
  }

  public ImmutableMap<PathFragment, RepositoryName> getManagedDirectories() {
    return managedDirectories;
  }

  public ImmutableSortedSet<String> getDoNotSymlinkInExecrootPaths() {
    return doNotSymlinkInExecrootPaths;
  }
}
