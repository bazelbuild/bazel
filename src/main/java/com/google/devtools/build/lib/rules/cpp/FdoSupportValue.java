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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.Path;
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
@Immutable
public class FdoSupportValue implements SkyValue {
  public static final SkyFunctionName SKYFUNCTION = SkyFunctionName.create("FDO_SUPPORT");

  /**
   * {@link SkyKey} for {@link FdoSupportValue}.
   */
  @Immutable
  public static class Key {
    private final LipoMode lipoMode;
    private final Path fdoZip;
    private final PathFragment fdoInstrument;

    private Key(LipoMode lipoMode, Path fdoZip, PathFragment fdoInstrument) {
      this.lipoMode = lipoMode;
      this.fdoZip = fdoZip;
      this.fdoInstrument = fdoInstrument;
    }

    public LipoMode getLipoMode() {
      return lipoMode;
    }

    public Path getFdoZip() {
      return fdoZip;
    }

    public PathFragment getFdoInstrument() {
      return fdoInstrument;
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
          && Objects.equals(this.fdoInstrument, that.fdoInstrument);
    }

    @Override
    public int hashCode() {
      return Objects.hash(lipoMode, fdoZip, fdoInstrument);
    }
  }

  private final FdoSupport fdoSupport;

  FdoSupportValue(FdoSupport fdoSupport) {
    this.fdoSupport = fdoSupport;
  }

  public FdoSupport getFdoSupport() {
    return fdoSupport;
  }

  public static SkyKey key(LipoMode lipoMode, Path fdoZip, PathFragment fdoInstrument) {
    return SkyKey.create(SKYFUNCTION, new Key(lipoMode, fdoZip, fdoInstrument));
  }
}
