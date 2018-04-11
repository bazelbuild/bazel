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
import com.google.devtools.build.lib.rules.cpp.FdoSupport.FdoMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * Wrapper for {@link FdoSupport}.
 *
 * <p>Things could probably be refactored such the attributes of {@link FdoSupport} are moved here
 * and the code in it to {@link FdoSupportFunction}. This would let us eliminate {@link FdoSupport}.
 *
 * <p>The eventual plan is to migrate FDO functionality to the execution phase once directory
 * artifacts work better, so this may not be worth it.
 */
@AutoCodec
@Immutable
public class FdoSupportValue implements SkyValue {
  public static final SkyFunctionName SKYFUNCTION = SkyFunctionName.create("FDO_SUPPORT");

  /** {@link SkyKey} for {@link FdoSupportValue}. */
  @Immutable
  @AutoCodec
  public static class Key implements SkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private final LipoMode lipoMode;
    private final PathFragment fdoZip;
    private final String fdoInstrument;
    private final FdoMode fdoMode;

    private Key(LipoMode lipoMode, PathFragment fdoZip, String fdoInstrument, FdoMode fdoMode) {
      this.lipoMode = lipoMode;
      this.fdoZip = fdoZip;
      this.fdoInstrument = fdoInstrument;
      this.fdoMode = fdoMode;
    }

    @AutoCodec.Instantiator
    @AutoCodec.VisibleForSerialization
    static Key of(LipoMode lipoMode, PathFragment fdoZip, String fdoInstrument, FdoMode fdoMode) {
      return interner.intern(new Key(lipoMode, fdoZip, fdoInstrument, fdoMode));
    }

    public LipoMode getLipoMode() {
      return lipoMode;
    }

    public PathFragment getFdoZip() {
      return fdoZip;
    }

    public String getFdoInstrument() {
      return fdoInstrument;
    }

    public FdoMode getFdoMode() {
      return fdoMode;
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
      return Objects.equals(this.lipoMode, that.lipoMode)
          && Objects.equals(this.fdoZip, that.fdoZip)
          && Objects.equals(this.fdoMode, that.fdoMode)
          && Objects.equals(this.fdoInstrument, that.fdoInstrument);
    }

    @Override
    public int hashCode() {
      return Objects.hash(lipoMode, fdoZip, fdoInstrument);
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

  public static SkyKey key(
      LipoMode lipoMode, PathFragment fdoZip, String fdoInstrument, FdoMode fdoMode) {
    return Key.of(lipoMode, fdoZip, fdoInstrument, fdoMode);
  }
}
