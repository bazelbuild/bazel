// Copyright 2015 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.skyframe.SkyFunctions.FILE_STATE;

import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.Set;

import javax.annotation.Nullable;

/** Utilities for checking dirtiness of keys (mainly filesystem keys) in the graph. */
class DirtinessCheckerUtils {
  private DirtinessCheckerUtils() {}

  static class BasicFilesystemDirtinessChecker implements SkyValueDirtinessChecker {
    protected boolean applies(SkyKey skyKey) {
      SkyFunctionName functionName = skyKey.functionName();
      return (functionName.equals(FILE_STATE) || functionName.equals(DIRECTORY_LISTING_STATE));
    }

    @Override
    @Nullable
    public DirtyResult maybeCheck(SkyKey skyKey, SkyValue skyValue,
        TimestampGranularityMonitor tsgm) {
      if (!applies(skyKey)) {
        return null;
      }
      RootedPath rootedPath = (RootedPath) skyKey.argument();
      if (skyKey.functionName().equals(FILE_STATE)) {
        return checkFileStateValue(rootedPath, (FileStateValue) skyValue, tsgm);
      } else {
        return checkDirectoryListingStateValue(rootedPath, (DirectoryListingStateValue) skyValue);
      }
    }

    private static DirtyResult checkFileStateValue(
        RootedPath rootedPath, FileStateValue fileStateValue, TimestampGranularityMonitor tsgm) {
      try {
        FileStateValue newValue = FileStateValue.create(rootedPath, tsgm);
        return newValue.equals(fileStateValue)
            ? DirtyResult.NOT_DIRTY
            : DirtyResult.dirtyWithNewValue(newValue);
      } catch (InconsistentFilesystemException | IOException e) {
        // TODO(bazel-team): An IOException indicates a failure to get a file digest or a symlink
        // target, not a missing file. Such a failure really shouldn't happen, so failing early
        // may be better here.
        return DirtyResult.DIRTY;
      }
    }

    private static DirtyResult checkDirectoryListingStateValue(
        RootedPath dirRootedPath, DirectoryListingStateValue directoryListingStateValue) {
      try {
        DirectoryListingStateValue newValue = DirectoryListingStateValue.create(dirRootedPath);
        return newValue.equals(directoryListingStateValue)
            ? DirtyResult.NOT_DIRTY
            : DirtyResult.dirtyWithNewValue(newValue);
      } catch (IOException e) {
        return DirtyResult.DIRTY;
      }
    }
  }

  static final class MissingDiffDirtinessChecker extends BasicFilesystemDirtinessChecker {
    private final Set<Path> missingDiffPaths;

    MissingDiffDirtinessChecker(final Set<Path> missingDiffPaths) {
      this.missingDiffPaths = missingDiffPaths;
    }

    @Override
    protected boolean applies(SkyKey skyKey) {
      return super.applies(skyKey)
          && missingDiffPaths.contains(((RootedPath) skyKey.argument()).getRoot());
    }
  }

  /** {@link SkyValueDirtinessChecker} that encompasses a union of other dirtiness checkers. */
  static final class UnionDirtinessChecker implements SkyValueDirtinessChecker {
    private final Iterable<SkyValueDirtinessChecker> dirtinessCheckers;

    UnionDirtinessChecker(Iterable<SkyValueDirtinessChecker> dirtinessCheckers) {
      this.dirtinessCheckers = dirtinessCheckers;
    }

    @Override
    @Nullable
    public DirtyResult maybeCheck(SkyKey key, SkyValue oldValue, TimestampGranularityMonitor tsgm) {
      for (SkyValueDirtinessChecker dirtinessChecker : dirtinessCheckers) {
        DirtyResult dirtyResult = dirtinessChecker.maybeCheck(key, oldValue, tsgm);
        if (dirtyResult != null) {
          return dirtyResult;
        }
      }
      return null;
    }
  }
}
