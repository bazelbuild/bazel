// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.errorprone.annotations.Immutable;
import com.google.errorprone.annotations.ThreadSafe;
import java.util.stream.Stream;

/** A value representing whether the casing of a {@link RootedPath} is canonical. */
@CheckReturnValue
@Immutable
@ThreadSafe
public sealed interface RootedPathCasingValue extends SkyValue {
  RootedPathCasingValue CANONICAL = new Canonical();

  @AutoCodec
  record Canonical() implements RootedPathCasingValue {}

  @AutoCodec
  record NonCanonical(RootedPathCasingValue parentValue, String canonicalBasename)
      implements RootedPathCasingValue {}

  default PathFragment expectedCasing(PathFragment rootRelativePath) {
    var fixedSegments =
        Stream.iterate(this, v -> v instanceof NonCanonical, v -> ((NonCanonical) v).parentValue)
            .map(v -> PathFragment.createAlreadyNormalized(((NonCanonical) v).canonicalBasename()))
            .reduce(PathFragment.EMPTY_FRAGMENT, (a, b) -> b.getRelative(a));
    return rootRelativePath
        .subFragment(0, rootRelativePath.segmentCount() - fixedSegments.segmentCount())
        .getRelative(fixedSegments);
  }

  static Key key(RootedPath path) {
    return Key.intern(new Key(path));
  }

  /** {@link SkyKey} implementation used for {@link RootedPathCasingFunction}. */
  @CheckReturnValue
  @ThreadSafe
  @AutoCodec
  final class Key extends AbstractSkyKey<RootedPath> {

    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    Key(RootedPath arg) {
      super(arg);
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ROOTED_PATH_CASING;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
