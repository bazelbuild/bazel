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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * The value computed by {@link CollectPackagesUnderDirectoryFunction}. Contains a mapping for all
 * its non-excluded directories to whether there are packages or error messages beneath them.
 *
 * <p>This value is used by {@link
 * com.google.devtools.build.lib.pkgcache.RecursivePackageProvider#streamPackagesUnderDirectory} to
 * help it traverse the graph and find the set of packages under a directory, recursively by {@link
 * CollectPackagesUnderDirectoryFunction} which computes a value for a directory by aggregating
 * results calculated from its subdirectories, and by {@link
 * PrepareDepsOfTargetsUnderDirectoryFunction} which uses this value to find transitive targets to
 * load.
 *
 * <p>Note that even though the {@link CollectPackagesUnderDirectoryFunction} is evaluated in part
 * because of its side-effects (i.e. loading transitive dependencies of targets), this value
 * interacts safely with change pruning, despite the fact that this value is a lossy representation
 * of the packages beneath a directory (i.e. it doesn't care <b>which</b> packages are under a
 * directory, just whether there are any). When the targets in a package change, the {@link
 * PackageValue} that {@link CollectPackagesUnderDirectoryFunction} depends on will be invalidated,
 * and the PrepareDeps function for that package's directory will be reevaluated, loading any new
 * transitive dependencies. Change pruning may prevent the reevaluation of PrepareDeps for
 * directories above that one, but they don't need to be re-run.
 */
public abstract class CollectPackagesUnderDirectoryValue implements SkyValue {
  @AutoCodec.VisibleForSerialization
  protected final ImmutableMap<RootedPath, Boolean>
      subdirectoryTransitivelyContainsPackagesOrErrors;

  CollectPackagesUnderDirectoryValue(
      ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackagesOrErrors) {
    this.subdirectoryTransitivelyContainsPackagesOrErrors =
        Preconditions.checkNotNull(subdirectoryTransitivelyContainsPackagesOrErrors);
  }

  /** Represents a successfully loaded package or a directory without a BUILD file. */
  @AutoCodec
  public static class NoErrorCollectPackagesUnderDirectoryValue
      extends CollectPackagesUnderDirectoryValue {
    @AutoCodec
    public static final NoErrorCollectPackagesUnderDirectoryValue EMPTY =
        new NoErrorCollectPackagesUnderDirectoryValue(
            false, ImmutableMap.<RootedPath, Boolean>of());

    private final boolean isDirectoryPackage;

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    NoErrorCollectPackagesUnderDirectoryValue(
        boolean isDirectoryPackage,
        ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackagesOrErrors) {
      super(subdirectoryTransitivelyContainsPackagesOrErrors);
      this.isDirectoryPackage = isDirectoryPackage;
    }

    @Override
    public boolean isDirectoryPackage() {
      return isDirectoryPackage;
    }

    @Nullable
    @Override
    public String getErrorMessage() {
      return null;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          isDirectoryPackage, getSubdirectoryTransitivelyContainsPackagesOrErrors());
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof NoErrorCollectPackagesUnderDirectoryValue)) {
        return false;
      }
      NoErrorCollectPackagesUnderDirectoryValue that =
          (NoErrorCollectPackagesUnderDirectoryValue) o;
      return this.isDirectoryPackage == that.isDirectoryPackage
          && Objects.equals(
              this.getSubdirectoryTransitivelyContainsPackagesOrErrors(),
              that.getSubdirectoryTransitivelyContainsPackagesOrErrors());
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("isDirectoryPackage", isDirectoryPackage)
          .add(
              "subdirectoryTransitivelyContainsPackagesOrErrors",
              getSubdirectoryTransitivelyContainsPackagesOrErrors())
          .toString();
    }

  }

  /** Represents a directory with a BUILD file that failed to load. */
  @AutoCodec
  public static class ErrorCollectPackagesUnderDirectoryValue
      extends CollectPackagesUnderDirectoryValue {
    private final String errorMessage;

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    ErrorCollectPackagesUnderDirectoryValue(
        String errorMessage,
        ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackagesOrErrors) {
      super(subdirectoryTransitivelyContainsPackagesOrErrors);
      this.errorMessage = Preconditions.checkNotNull(errorMessage);
    }

    @Override
    public boolean isDirectoryPackage() {
      return false;
    }

    @Override
    public String getErrorMessage() {
      return errorMessage;
    }

    @Override
    public int hashCode() {
      return Objects.hash(errorMessage, getSubdirectoryTransitivelyContainsPackagesOrErrors());
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof ErrorCollectPackagesUnderDirectoryValue)) {
        return false;
      }
      ErrorCollectPackagesUnderDirectoryValue that = (ErrorCollectPackagesUnderDirectoryValue) o;
      return Objects.equals(this.errorMessage, that.errorMessage)
          && Objects.equals(
              this.getSubdirectoryTransitivelyContainsPackagesOrErrors(),
              that.getSubdirectoryTransitivelyContainsPackagesOrErrors());
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("errorMessage", errorMessage)
          .add(
              "subdirectoryTransitivelyContainsPackagesOrErrors",
              getSubdirectoryTransitivelyContainsPackagesOrErrors())
          .toString();
    }
  }

  /**
   * Constructs a {@link CollectPackagesUnderDirectoryValue} for a directory with a BUILD file that
   * failed to load as a package.
   */
  public static CollectPackagesUnderDirectoryValue ofError(
      String errorMessage,
      ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackagesOrErrors) {
    Preconditions.checkNotNull(errorMessage, "errorMessage");
    return new ErrorCollectPackagesUnderDirectoryValue(
        errorMessage, subdirectoryTransitivelyContainsPackagesOrErrors);
  }

  /**
   * Constructs a {@link CollectPackagesUnderDirectoryValue} for a directory without a BUILD file or
   * that has a BUILD file that successfully loads as a package.
   */
  public static CollectPackagesUnderDirectoryValue ofNoError(
      boolean isDirectoryPackage,
      ImmutableMap<RootedPath, Boolean> subdirectoryTransitivelyContainsPackagesOrErrors) {
    if (!isDirectoryPackage && subdirectoryTransitivelyContainsPackagesOrErrors.isEmpty()) {
      return NoErrorCollectPackagesUnderDirectoryValue.EMPTY;
    }
    return new NoErrorCollectPackagesUnderDirectoryValue(
        isDirectoryPackage, subdirectoryTransitivelyContainsPackagesOrErrors);
  }

  /**
   * Returns whether there is a BUILD file in this directory that can be loaded as a package. If
   * this returns {@code true}, then {@link #getErrorMessage()} returns {@code null}.
   */
  public abstract boolean isDirectoryPackage();

  /**
   * Returns an error describing why the BUILD file in this directory cannot be loaded as a package,
   * if there is one and it can't be. Otherwise returns {@code null}. If this returns non-{@code
   * null}, then {@link #isDirectoryPackage()} returns {@code false}.
   */
  @Nullable
  public abstract String getErrorMessage();

  /**
   * Returns an {@link ImmutableMap} describing each immediate subdirectory of this directory and
   * whether there are any packages, or BUILD files that couldn't be loaded, in or beneath that
   * subdirectory.
   */
  public final ImmutableMap<RootedPath, Boolean>
      getSubdirectoryTransitivelyContainsPackagesOrErrors() {
    return subdirectoryTransitivelyContainsPackagesOrErrors;
  }

  /** Create a collect packages under directory request. */
  @ThreadSafe
  public static SkyKey key(
      RepositoryName repository, RootedPath rootedPath, ImmutableSet<PathFragment> excludedPaths) {
    return Key.create(repository, rootedPath, excludedPaths);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends RecursivePkgSkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(
        RepositoryName repositoryName,
        RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      super(repositoryName, rootedPath, excludedPaths);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(
        RepositoryName repositoryName,
        RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      return interner.intern(new Key(repositoryName, rootedPath, excludedPaths));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.COLLECT_PACKAGES_UNDER_DIRECTORY;
    }
  }
}
