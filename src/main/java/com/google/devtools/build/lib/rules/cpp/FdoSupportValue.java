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
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * Wrapper for {@link FdoSupport}.
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

    private final PathFragment fdoZip;

    private Key(PathFragment fdoZip) {
      this.fdoZip = fdoZip;
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(PathFragment fdoZip) {
      return interner.intern(new Key(fdoZip));
    }

    public PathFragment getFdoZip() {
      return fdoZip;
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
      return Objects.equals(this.fdoZip, that.fdoZip);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fdoZip);
    }

    @Override
    public SkyFunctionName functionName() {
      return SKYFUNCTION;
    }
  }

  private final FdoSupport fdoSupport;

  FdoSupportValue(FdoSupport fdoSupport) {
    this.fdoSupport = fdoSupport;
  }

  public FdoSupport getFdoSupport() {
    return fdoSupport;
  }

  public static SkyKey key(PathFragment fdoZip) {
    return Key.of(fdoZip);
  }
}
