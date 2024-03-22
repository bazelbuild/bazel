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
package com.google.devtools.build.lib.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A value that corresponds to a file (or directory or symlink or non-existent file), fully
 * accounting for symlinks (e.g. proper dependencies on ancestor symlinks so as to be incrementally
 * correct). Anything in Skyframe that cares about the fully resolved path of a file (e.g. anything
 * that cares about the contents of a file) should have a dependency on the corresponding {@link
 * FileValue}.
 *
 * <p>Note that the existence of a file value does not imply that the file exists on the filesystem.
 * File values for missing files will be created on purpose in order to facilitate incremental
 * builds in the case those files have reappeared.
 *
 * <p>This class contains the relevant metadata for a file, although not the contents. Note that
 * since a FileValue doesn't store its corresponding SkyKey, it's possible for the FileValues for
 * two different paths to be the same.
 *
 * <p>This should not be used for build outputs; use {@link ArtifactSkyKey} to create keys for
 * those.
 */
@Immutable
@ThreadSafe
public abstract class FileValue implements SkyValue {
  // Depends non-hermetically on package path, but that is under the control of a flag, so use
  // semi-hermetic.
  public static final SkyFunctionName FILE = SkyFunctionName.createSemiHermetic("FILE");

  public boolean exists() {
    return realFileStateValue().getType() != FileStateType.NONEXISTENT;
  }

  /** Returns true if the original path is a symlink; the target path can never be a symlink. */
  public boolean isSymlink() {
    return false;
  }

  /**
   * Returns true if this value corresponds to a file or symlink to an existing regular or special
   * file. If so, its parent directory is guaranteed to exist.
   */
  public boolean isFile() {
    return realFileStateValue().getType() == FileStateType.REGULAR_FILE
        || realFileStateValue().getType() == FileStateType.SPECIAL_FILE;
  }

  /**
   * Returns true if this value corresponds to a special file or symlink to a special file. If so,
   * its parent directory is guaranteed to exist.
   */
  public boolean isSpecialFile() {
    return realFileStateValue().getType() == FileStateType.SPECIAL_FILE;
  }

  /**
   * Returns true if the file is a directory or a symlink to an existing directory. If so, its
   * parent directory is guaranteed to exist.
   */
  public boolean isDirectory() {
    return realFileStateValue().getType() == FileStateType.DIRECTORY;
  }

  /**
   * If {@code !isFile() && exists()}, returns an ordered list of the {@link RootedPath}s that were
   * considered when determining {@code realRootedPath()}.
   *
   * <p>This information is used to detect unbounded symlink expansions.
   *
   * <p>As a memory optimization, we don't store this information when {@code isFile() || !exists()}
   * -- this information is only needed for resolving ancestors, and an existing file or a
   * non-existent directory has no descendants, by definition.
   */
  public abstract ImmutableList<RootedPath> logicalChainDuringResolution();

  /**
   * If a symlink pointing back to its own ancestor was encountered during the resolution of this
   * {@link FileValue}, returns the path to it. Otherwise, returns null.
   */
  public abstract ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain();

  /**
   * If a symlink pointing back to its own ancestor was encountered during the resolution of this
   * {@link FileValue}, returns the symlinks in the cycle. Otherwise, returns null.
   */
  public abstract ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain();

  /**
   * Returns the real rooted path of the file, taking ancestor symlinks into account. For example,
   * the rooted path ['root']/['a/b'] is really ['root']/['c/b'] if 'a' is a symlink to 'c'. Note
   * that ancestor symlinks outside the root boundary are not taken into consideration.
   */
  public abstract RootedPath realRootedPath();

  public abstract FileStateValue realFileStateValue();

  /**
   * Returns the unresolved link target if {@link #isSymlink()}.
   *
   * <p>This is useful if the caller wants to, for example, duplicate a relative symlink. An actual
   * example could be a build rule that copies a set of input files to the output directory, but
   * upon encountering symbolic links it can decide between copying or following them.
   */
  public PathFragment getUnresolvedLinkTarget() {
    throw new IllegalStateException(this.toString());
  }

  public long getSize() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getSize();
  }

  @Nullable
  public byte[] getDigest() {
    Preconditions.checkState(isFile(), this);
    return realFileStateValue().getDigest();
  }

  /** Returns a key for building a file value for the given root-relative path. */
  @ThreadSafe
  public static Key key(RootedPath rootedPath) {
    return Key.create(rootedPath);
  }

  /** Key type for FileValue. */
  @VisibleForSerialization
  @AutoCodec
  public static class Key extends AbstractSkyKey<RootedPath> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RootedPath arg) {
      super(arg);
    }

    private static Key create(RootedPath arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return FILE;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  /**
   * Only intended to be used by {@link com.google.devtools.build.lib.skyframe.FileFunction}. Should
   * not be used for symlink cycles.
   */
  public static FileValue value(
      ImmutableList<RootedPath> logicalChainDuringResolution,
      ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain,
      ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain,
      RootedPath originalRootedPath,
      FileStateValue fileStateValueFromAncestors,
      RootedPath realRootedPath,
      FileStateValue realFileStateValue) {
    if (originalRootedPath.equals(realRootedPath)) {
      Preconditions.checkState(
          fileStateValueFromAncestors.getType() != FileStateType.SYMLINK,
          "originalRootedPath: %s, fileStateValueFromAncestors: %s, "
              + "realRootedPath: %s, fileStateValueFromAncestors: %s",
          originalRootedPath,
          fileStateValueFromAncestors,
          realRootedPath,
          realFileStateValue);
      Preconditions.checkState(
          !realFileStateValue.getType().exists()
              || realFileStateValue.getType().isFile()
              || Iterables.getOnlyElement(logicalChainDuringResolution).equals(originalRootedPath),
          "logicalChainDuringResolution: %s, originalRootedPath: %s",
          logicalChainDuringResolution,
          originalRootedPath);
      return new RegularFileValue(originalRootedPath, fileStateValueFromAncestors);
    }

    boolean shouldStoreChain =
        switch (realFileStateValue.getType()) {
          case REGULAR_FILE, SPECIAL_FILE, NONEXISTENT -> false;
          case SYMLINK, DIRECTORY -> true;
        };

    if (fileStateValueFromAncestors.getType() == FileStateType.SYMLINK) {
      PathFragment symlinkTarget = fileStateValueFromAncestors.getSymlinkTarget();
      if (pathToUnboundedAncestorSymlinkExpansionChain != null) {
        return new SymlinkFileValueWithSymlinkCycle(
            realRootedPath,
            realFileStateValue,
            logicalChainDuringResolution,
            symlinkTarget,
            pathToUnboundedAncestorSymlinkExpansionChain,
            unboundedAncestorSymlinkExpansionChain);
      } else if (shouldStoreChain) {
        return new SymlinkFileValueWithStoredChain(
            realRootedPath, realFileStateValue, logicalChainDuringResolution, symlinkTarget);
      } else {
        return new SymlinkFileValueWithoutStoredChain(
            realRootedPath, realFileStateValue, symlinkTarget);
      }
    } else {
      if (pathToUnboundedAncestorSymlinkExpansionChain != null) {
        return new DifferentRealPathFileValueWithSymlinkCycle(
            realRootedPath,
            realFileStateValue,
            logicalChainDuringResolution,
            pathToUnboundedAncestorSymlinkExpansionChain,
            unboundedAncestorSymlinkExpansionChain);
      } else if (shouldStoreChain) {
        return new DifferentRealPathFileValueWithStoredChain(
            realRootedPath, realFileStateValue, logicalChainDuringResolution);
      } else {
        return new DifferentRealPathFileValueWithoutStoredChain(realRootedPath, realFileStateValue);
      }
    }
  }

  /**
   * Implementation of {@link FileValue} for paths whose fully resolved path is the same as the
   * requested path. For example, this is the case for the path "foo/bar/baz" if neither 'foo' nor
   * 'foo/bar' nor 'foo/bar/baz' are symlinks.
   */
  @VisibleForTesting
  public static final class RegularFileValue extends FileValue {

    private final RootedPath rootedPath;
    private final FileStateValue fileStateValue;

    @VisibleForTesting
    public RegularFileValue(RootedPath rootedPath, FileStateValue fileStateValue) {
      this.rootedPath = Preconditions.checkNotNull(rootedPath);
      this.fileStateValue = Preconditions.checkNotNull(fileStateValue);
    }

    @Override
    public ImmutableList<RootedPath> logicalChainDuringResolution() {
      return ImmutableList.of(rootedPath);
    }

    @Nullable
    @Override
    public ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain() {
      return null;
    }

    @Nullable
    @Override
    public ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain() {
      return null;
    }

    @Override
    public RootedPath realRootedPath() {
      return rootedPath;
    }

    @Override
    public FileStateValue realFileStateValue() {
      return fileStateValue;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (!(obj instanceof RegularFileValue)) {
        return false;
      }
      RegularFileValue other = (RegularFileValue) obj;
      return rootedPath.equals(other.rootedPath) && fileStateValue.equals(other.fileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(rootedPath, fileStateValue);
    }

    @Override
    public String toString() {
      return String.format("non-symlink (path=%s, state=%s)", rootedPath, fileStateValue);
    }
  }

  /**
   * A {@link FileValue} whose resolution required traversing a symlink chain caused by a symlink
   * pointing to its own ancestor but which eventually points to a real file.
   */
  private static class DifferentRealPathFileValueWithSymlinkCycle
      extends DifferentRealPathFileValueWithStoredChain {
    // We can't store an exception here because this needs to be serialized, AutoCodec chokes on
    // object cycles and FilesystemInfiniteSymlinkCycleException somehow sets its cause to itself
    protected final ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain;
    protected final ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain;

    DifferentRealPathFileValueWithSymlinkCycle(
        RootedPath realRootedPath,
        FileStateValue realFileStateValue,
        ImmutableList<RootedPath> logicalChainDuringResolution,
        ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain,
        ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain) {
      super(realRootedPath, realFileStateValue, logicalChainDuringResolution);
      this.pathToUnboundedAncestorSymlinkExpansionChain =
          pathToUnboundedAncestorSymlinkExpansionChain;
      this.unboundedAncestorSymlinkExpansionChain = unboundedAncestorSymlinkExpansionChain;
    }

    @Override
    public ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain() {
      return pathToUnboundedAncestorSymlinkExpansionChain;
    }

    @Override
    public ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain() {
      return unboundedAncestorSymlinkExpansionChain;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          realRootedPath,
          realFileStateValue,
          logicalChainDuringResolution,
          pathToUnboundedAncestorSymlinkExpansionChain,
          unboundedAncestorSymlinkExpansionChain);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }

      if (obj.getClass() != DifferentRealPathFileValueWithSymlinkCycle.class) {
        return false;
      }

      DifferentRealPathFileValueWithSymlinkCycle other =
          (DifferentRealPathFileValueWithSymlinkCycle) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && logicalChainDuringResolution.equals(other.logicalChainDuringResolution)
          && pathToUnboundedAncestorSymlinkExpansionChain.equals(
              other.pathToUnboundedAncestorSymlinkExpansionChain)
          && unboundedAncestorSymlinkExpansionChain.equals(
              other.unboundedAncestorSymlinkExpansionChain);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink ancestor (real_path=%s, real_state=%s, chain=%s, path=%s, cycle=%s)",
          realRootedPath,
          realFileStateValue,
          logicalChainDuringResolution,
          pathToUnboundedAncestorSymlinkExpansionChain,
          unboundedAncestorSymlinkExpansionChain);
    }
  }

  /**
   * Implementation of {@link FileValue} for paths whose fully resolved path is different than the
   * requested path, but the path itself is not a symlink. For example, this is the case for the
   * path "foo/bar/baz" if at least one of {'foo', 'foo/bar'} is a symlink but 'foo/bar/baz' not.
   */
  @VisibleForTesting
  public static class DifferentRealPathFileValueWithStoredChain extends FileValue {
    protected final RootedPath realRootedPath;
    protected final FileStateValue realFileStateValue;
    protected final ImmutableList<RootedPath> logicalChainDuringResolution;

    @VisibleForTesting
    public DifferentRealPathFileValueWithStoredChain(
        RootedPath realRootedPath,
        FileStateValue realFileStateValue,
        ImmutableList<RootedPath> logicalChainDuringResolution) {
      this.realRootedPath = Preconditions.checkNotNull(realRootedPath);
      this.realFileStateValue = Preconditions.checkNotNull(realFileStateValue);
      this.logicalChainDuringResolution = logicalChainDuringResolution;
    }

    @Override
    public RootedPath realRootedPath() {
      return realRootedPath;
    }

    @Override
    public FileStateValue realFileStateValue() {
      return realFileStateValue;
    }

    @Override
    public ImmutableList<RootedPath> logicalChainDuringResolution() {
      return logicalChainDuringResolution;
    }

    @Override
    public ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain() {
      return null;
    }

    @Override
    public ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain() {
      return null;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      // Note that we can't use 'instanceof' because this class has a subclass.
      if (obj.getClass() != DifferentRealPathFileValueWithStoredChain.class) {
        return false;
      }
      DifferentRealPathFileValueWithStoredChain other =
          (DifferentRealPathFileValueWithStoredChain) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && logicalChainDuringResolution.equals(other.logicalChainDuringResolution);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realRootedPath, realFileStateValue, logicalChainDuringResolution);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink ancestor (real_path=%s, real_state=%s, chain=%s)",
          realRootedPath, realFileStateValue, logicalChainDuringResolution);
    }
  }

  /**
   * Same as {@link DifferentRealPathFileValueWithStoredChain}, except without {@link
   * #logicalChainDuringResolution}.
   */
  @VisibleForTesting
  public static class DifferentRealPathFileValueWithoutStoredChain extends FileValue {
    protected final RootedPath realRootedPath;
    protected final FileStateValue realFileStateValue;

    @VisibleForTesting
    public DifferentRealPathFileValueWithoutStoredChain(
        RootedPath realRootedPath, FileStateValue realFileStateValue) {
      this.realRootedPath = Preconditions.checkNotNull(realRootedPath);
      this.realFileStateValue = Preconditions.checkNotNull(realFileStateValue);
    }

    @Override
    public RootedPath realRootedPath() {
      return realRootedPath;
    }

    @Override
    public FileStateValue realFileStateValue() {
      return realFileStateValue;
    }

    @Override
    public ImmutableList<RootedPath> logicalChainDuringResolution() {
      throw new IllegalStateException(this.toString());
    }

    @Override
    public ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain() {
      return null;
    }

    @Override
    public ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain() {
      return null;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      // Note that we can't use 'instanceof' because this class has a subclass.
      if (obj.getClass() != DifferentRealPathFileValueWithoutStoredChain.class) {
        return false;
      }
      DifferentRealPathFileValueWithoutStoredChain other =
          (DifferentRealPathFileValueWithoutStoredChain) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realRootedPath, realFileStateValue);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink ancestor (real_path=%s, real_state=%s)", realRootedPath, realFileStateValue);
    }
  }

  /**
   * A {@link FileValue} whose resolution required traversing a symlink chain caused by a symlink
   * pointing to its own ancestor and which eventually points to a symlink.
   */
  private static final class SymlinkFileValueWithSymlinkCycle
      extends SymlinkFileValueWithStoredChain {
    // We can't store an exception here because this needs to be serialized, AutoCodec chokes on
    // object cycles and FilesystemInfiniteSymlinkCycleException somehow sets its cause to itself
    private final ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain;
    private final ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain;

    SymlinkFileValueWithSymlinkCycle(
        RootedPath realRootedPath,
        FileStateValue realFileStateValue,
        ImmutableList<RootedPath> logicalChainDuringResolution,
        PathFragment linkTarget,
        ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain,
        ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain) {
      super(realRootedPath, realFileStateValue, logicalChainDuringResolution, linkTarget);
      this.pathToUnboundedAncestorSymlinkExpansionChain =
          pathToUnboundedAncestorSymlinkExpansionChain;
      this.unboundedAncestorSymlinkExpansionChain = unboundedAncestorSymlinkExpansionChain;
    }

    @Override
    public ImmutableList<RootedPath> pathToUnboundedAncestorSymlinkExpansionChain() {
      return pathToUnboundedAncestorSymlinkExpansionChain;
    }

    @Override
    public ImmutableList<RootedPath> unboundedAncestorSymlinkExpansionChain() {
      return unboundedAncestorSymlinkExpansionChain;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          realRootedPath,
          realFileStateValue,
          logicalChainDuringResolution,
          linkTarget,
          pathToUnboundedAncestorSymlinkExpansionChain,
          unboundedAncestorSymlinkExpansionChain);
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }

      if (obj.getClass() != SymlinkFileValueWithSymlinkCycle.class) {
        return false;
      }

      SymlinkFileValueWithSymlinkCycle other = (SymlinkFileValueWithSymlinkCycle) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && logicalChainDuringResolution.equals(other.logicalChainDuringResolution)
          && linkTarget.equals(other.linkTarget)
          && pathToUnboundedAncestorSymlinkExpansionChain.equals(
              other.pathToUnboundedAncestorSymlinkExpansionChain)
          && unboundedAncestorSymlinkExpansionChain.equals(
              other.unboundedAncestorSymlinkExpansionChain);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink ancestor (real_path=%s, real_state=%s, target=%s, chain=%s, path=%s, cycle=%s)",
          realRootedPath,
          realFileStateValue,
          linkTarget,
          logicalChainDuringResolution,
          pathToUnboundedAncestorSymlinkExpansionChain,
          unboundedAncestorSymlinkExpansionChain);
    }
  }

  /** Implementation of {@link FileValue} for paths that are themselves symlinks. */
  @VisibleForTesting
  public static class SymlinkFileValueWithStoredChain
      extends DifferentRealPathFileValueWithStoredChain {
    protected final PathFragment linkTarget;

    @VisibleForTesting
    public SymlinkFileValueWithStoredChain(
        RootedPath realRootedPath,
        FileStateValue realFileStateValue,
        ImmutableList<RootedPath> logicalChainDuringResolution,
        PathFragment linkTarget) {
      super(realRootedPath, realFileStateValue, logicalChainDuringResolution);
      this.linkTarget = linkTarget;
    }

    @Override
    public boolean isSymlink() {
      return true;
    }

    @Override
    public PathFragment getUnresolvedLinkTarget() {
      return linkTarget;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (!(obj instanceof SymlinkFileValueWithStoredChain)) {
        return false;
      }
      SymlinkFileValueWithStoredChain other = (SymlinkFileValueWithStoredChain) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && logicalChainDuringResolution.equals(other.logicalChainDuringResolution)
          && linkTarget.equals(other.linkTarget);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          realRootedPath, realFileStateValue, logicalChainDuringResolution, linkTarget);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink (real_path=%s, real_state=%s, link_value=%s, chain=%s)",
          realRootedPath, realFileStateValue, linkTarget, logicalChainDuringResolution);
    }
  }

  /**
   * Same as {@link SymlinkFileValueWithStoredChain}, except without {@link
   * #logicalChainDuringResolution}.
   */
  @VisibleForTesting
  public static final class SymlinkFileValueWithoutStoredChain
      extends DifferentRealPathFileValueWithoutStoredChain {
    private final PathFragment linkTarget;

    @VisibleForTesting
    public SymlinkFileValueWithoutStoredChain(
        RootedPath realRootedPath, FileStateValue realFileStateValue, PathFragment linkTarget) {
      super(realRootedPath, realFileStateValue);
      this.linkTarget = linkTarget;
    }

    @Override
    public boolean isSymlink() {
      return true;
    }

    @Override
    public PathFragment getUnresolvedLinkTarget() {
      return linkTarget;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == null) {
        return false;
      }
      if (!(obj instanceof SymlinkFileValueWithoutStoredChain)) {
        return false;
      }
      SymlinkFileValueWithoutStoredChain other = (SymlinkFileValueWithoutStoredChain) obj;
      return realRootedPath.equals(other.realRootedPath)
          && realFileStateValue.equals(other.realFileStateValue)
          && linkTarget.equals(other.linkTarget);
    }

    @Override
    public int hashCode() {
      return Objects.hash(realRootedPath, realFileStateValue, linkTarget);
    }

    @Override
    public String toString() {
      return String.format(
          "symlink (real_path=%s, real_state=%s, link_value=%s)",
          realRootedPath, realFileStateValue, linkTarget);
    }
  }
}
