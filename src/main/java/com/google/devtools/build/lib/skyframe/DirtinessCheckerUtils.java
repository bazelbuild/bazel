// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.skyframe.SkyFunctions.DIRECTORY_LISTING_STATE;
import static com.google.devtools.build.lib.vfs.FileStateKey.FILE_STATE;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.FileType;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.Version;
import java.io.IOException;
import java.util.EnumSet;
import java.util.Set;
import javax.annotation.Nullable;

/** Utilities for checking dirtiness of keys (mainly filesystem keys) in the graph. */
public class DirtinessCheckerUtils {
  private DirtinessCheckerUtils() {}

  /** Checks dirtiness of file keys in the graph. */
  public static class FileDirtinessChecker extends SkyValueDirtinessChecker {
    @Override
    public boolean applies(SkyKey skyKey) {
      return skyKey.functionName().equals(FILE_STATE);
    }

    @Override
    @Nullable
    public SkyValue createNewValue(
        SkyKey key, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      return FileStateValue.create((RootedPath) key.argument(), syscallCache, tsgm);
    }
  }

  /** Checks dirtiness of directory keys in the graph. */
  public static class DirectoryDirtinessChecker extends SkyValueDirtinessChecker {
    @Override
    public boolean applies(SkyKey skyKey) {
      return skyKey.functionName().equals(DIRECTORY_LISTING_STATE);
    }

    @Override
    public SkyValue createNewValue(
        SkyKey key, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      RootedPath rootedPath = (RootedPath) key.argument();
      return DirectoryListingStateValue.create(syscallCache.readdir(rootedPath.asPath()));
    }
  }

  static final class MissingDiffDirtinessChecker extends SkyValueDirtinessChecker {
    private final Set<Root> missingDiffPackageRoots;
    private final UnionDirtinessChecker checker = createBasicFilesystemDirtinessChecker();

    MissingDiffDirtinessChecker(Set<Root> missingDiffPackageRoots) {
      this.missingDiffPackageRoots = missingDiffPackageRoots;
    }

    @Override
    public boolean applies(SkyKey key) {
      return checker.applies(key)
          && missingDiffPackageRoots.contains(((RootedPath) key.argument()).getRoot());
    }

    @Override
    public SkyValue createNewValue(
        SkyKey key, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      return checker.createNewValue(key, syscallCache, tsgm);
    }
  }

  /**
   * Serves for tracking whether there are external and output files {@see ExternalFilesKnowledge}.
   * Filtering of files, for which the new values should not be injected into evaluator, is done in
   * SequencedSkyframeExecutor.handleChangedFiles().
   */
  static final class ExternalDirtinessChecker extends SkyValueDirtinessChecker {
    private final ExternalFilesHelper externalFilesHelper;
    private final EnumSet<FileType> fileTypesToCheck;

    private final UnionDirtinessChecker checker = createBasicFilesystemDirtinessChecker();

    ExternalDirtinessChecker(ExternalFilesHelper externalFilesHelper,
        EnumSet<FileType> fileTypesToCheck) {
      this.externalFilesHelper = externalFilesHelper;
      this.fileTypesToCheck = fileTypesToCheck;
    }

    @Override
    public boolean applies(SkyKey key) {
      if (!checker.applies(key)) {
        return false;
      }
      FileType fileType = externalFilesHelper.getAndNoteFileType((RootedPath) key.argument());
      return fileTypesToCheck.contains(fileType);
    }

    @Override
    public SkyValue createNewValue(
        SkyKey key, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm) {
      throw new UnsupportedOperationException();
    }

    @Override
    public SkyValueDirtinessChecker.DirtyResult check(
        SkyKey skyKey,
        SkyValue oldValue,
        @Nullable Version oldMtsv,
        SyscallCache syscallCache,
        @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      FileType fileType = externalFilesHelper.getAndNoteFileType((RootedPath) skyKey.argument());
      boolean cacheable = isCacheableType(fileType);
      SkyValue newValue =
          checker.createNewValue(skyKey, cacheable ? syscallCache : SyscallCache.NO_CACHE, tsgm);
      if (Objects.equal(newValue, oldValue)) {
        return SkyValueDirtinessChecker.DirtyResult.notDirty();
      }
      if (cacheable) {
        return SkyValueDirtinessChecker.DirtyResult.dirtyWithNewValue(newValue);
      }
      // Files under output_base/external have a dependency on the WORKSPACE file, so we don't add
      // a new SkyValue to the graph yet because it might change once the WORKSPACE file has been
      // parsed. Similarly, output files might change during execution.
      return SkyValueDirtinessChecker.DirtyResult.dirty();
    }

    private static boolean isCacheableType(FileType fileType) {
      switch (fileType) {
        case INTERNAL:
        case EXTERNAL:
        case BUNDLED:
          return true;
        case EXTERNAL_REPO:
        case OUTPUT:
          return false;
      }
      throw new AssertionError("Unknown type " + fileType);
    }
  }

  static UnionDirtinessChecker createBasicFilesystemDirtinessChecker() {
    return new UnionDirtinessChecker(
        ImmutableList.of(new FileDirtinessChecker(), new DirectoryDirtinessChecker()));
  }

  /** {@link SkyValueDirtinessChecker} that encompasses a union of other dirtiness checkers. */
  public static final class UnionDirtinessChecker extends SkyValueDirtinessChecker {
    private final Iterable<SkyValueDirtinessChecker> dirtinessCheckers;

    public UnionDirtinessChecker(Iterable<SkyValueDirtinessChecker> dirtinessCheckers) {
      this.dirtinessCheckers = dirtinessCheckers;
    }

    @Nullable
    private SkyValueDirtinessChecker getChecker(SkyKey key) {
      for (SkyValueDirtinessChecker dirtinessChecker : dirtinessCheckers) {
        if (dirtinessChecker.applies(key)) {
          return dirtinessChecker;
        }
      }
      return null;
    }

    @Override
    public boolean applies(SkyKey key) {
      return getChecker(key) != null;
    }

    @Override
    public SkyValue createNewValue(
        SkyKey key, SyscallCache syscallCache, @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      return Preconditions.checkNotNull(getChecker(key), key)
          .createNewValue(key, syscallCache, tsgm);
    }

    @Override
    public DirtyResult check(
        SkyKey key,
        @Nullable SkyValue oldValue,
        @Nullable Version oldMtsv,
        SyscallCache syscallCache,
        @Nullable TimestampGranularityMonitor tsgm)
        throws IOException {
      return Preconditions.checkNotNull(getChecker(key), key)
          .check(key, oldValue, oldMtsv, syscallCache, tsgm);
    }

    @Override
    @Nullable
    public Version getMaxTransitiveSourceVersionForNewValue(SkyKey key, SkyValue value)
        throws IOException {
      return Preconditions.checkNotNull(getChecker(key), key)
          .getMaxTransitiveSourceVersionForNewValue(key, value);
    }
  }
}
