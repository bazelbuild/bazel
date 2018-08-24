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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A container for the path to the FDO profile.
 *
 * <p>{@link FdoSupportValue} is created from {@link FdoSupportFunction} (a {@link SkyFunction}),
 * which is requested from Skyframe by the {@code cc_toolchain} rule. It's done this way because
 * the path depends on both a command line argument and the location of the workspace and the latter
 * is not available either during configuration creation or during the analysis phase.
 */
@AutoCodec
@Immutable
public class FdoSupportValue implements SkyValue {
  public static final SkyFunctionName SKYFUNCTION = SkyFunctionName.createHermetic("FDO_SUPPORT");

  /** {@link SkyKey} for {@link FdoSupportValue}. */
  @Immutable
  @AutoCodec
  public static class Key implements SkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private final PathFragment fdoProfileArgument;

    private Key(PathFragment fdoProfileArgument) {
      this.fdoProfileArgument = fdoProfileArgument;
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(PathFragment fdoProfileArgument) {
      return interner.intern(new Key(fdoProfileArgument));
    }

    public PathFragment getFdoProfileArgument() {
      return fdoProfileArgument;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }

      if (!(o instanceof Key)) {
        return false;
      }

      Key that = (Key) o;
      return Objects.equals(this.fdoProfileArgument, that.fdoProfileArgument);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fdoProfileArgument);
    }

    @Override
    public SkyFunctionName functionName() {
      return SKYFUNCTION;
    }
  }

  /**
   * Path of the profile file passed to {@code --fdo_optimize}
   */
  // TODO(lberki): This should be a PathFragment.
  // Except that CcProtoProfileProvider#getProfile() calls #exists() on it, which is ridiculously
  // incorrect.
  private final Path fdoProfile;

  FdoSupportValue(Path fdoProfile) {
    this.fdoProfile = fdoProfile;
  }

  public Path getFdoProfile() {
    return fdoProfile;
  }

  public static SkyKey key(PathFragment fdoProfileArgument) {
    return Key.of(fdoProfileArgument);
  }
}
