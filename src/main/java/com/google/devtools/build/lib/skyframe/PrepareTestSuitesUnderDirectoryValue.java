// Copyright 2018 The Bazel Authors. All rights reserved.
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

/** Dummy {@link SkyValue} for {@link PrepareDepsOfTargetsUnderDirectoryFunction}. */
public class PrepareTestSuitesUnderDirectoryValue implements SkyValue {
  @AutoCodec
  public static final PrepareTestSuitesUnderDirectoryValue INSTANCE =
      new PrepareTestSuitesUnderDirectoryValue();

  @ThreadSafe
  public static SkyKey key(
      RepositoryName repository, RootedPath rootedPath, ImmutableSet<PathFragment> excludedPaths) {
    return PrepareTestSuitesUnderDirectoryValue.Key.create(repository, rootedPath, excludedPaths);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends RecursivePkgSkyKey {
    private static final Interner<PrepareTestSuitesUnderDirectoryValue.Key> interner =
        BlazeInterners.newWeakInterner();

    private Key(
        RepositoryName repositoryName,
        RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      super(repositoryName, rootedPath, excludedPaths);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static PrepareTestSuitesUnderDirectoryValue.Key create(
        RepositoryName repositoryName,
        RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      return interner.intern(
          new PrepareTestSuitesUnderDirectoryValue.Key(repositoryName, rootedPath, excludedPaths));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PREPARE_TEST_SUITES_UNDER_DIRECTORY;
    }
  }
}
