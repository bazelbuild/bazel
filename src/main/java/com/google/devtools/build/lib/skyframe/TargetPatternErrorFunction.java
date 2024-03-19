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

import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * SkyFunction that throws a {@link TargetParsingException} for target pattern that could not be
 * parsed. Must only be requested when a SkyFunction wishes to ignore the errors in a target pattern
 * in keep_going mode, but to shut down the build in nokeep_going mode.
 *
 * <p>This SkyFunction never returns a value, only throws a {@link TargetParsingException}, and
 * should never return null, since all of its dependencies should already be present.
 */
public class TargetPatternErrorFunction implements SkyFunction {
  public static Key key(TargetParsingException e) {
    return Key.create(e.getMessage(), e.getDetailedExitCode());
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();
    private final String message;
    private final DetailedExitCode detailedExitCode;

    private Key(String message, DetailedExitCode detailedExitCode) {
      this.message = message;
      this.detailedExitCode = detailedExitCode;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(String message, DetailedExitCode detailedExitCode) {
      return interner.intern(new Key(message, detailedExitCode));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_PATTERN_ERROR;
    }

    @Override
    public int hashCode() {
      return 43 * message.hashCode() + detailedExitCode.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key)) {
        return false;
      }
      Key that = (Key) obj;
      return this.message.equals(that.message)
          && this.detailedExitCode.equals(that.detailedExitCode);
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws TargetErrorFunctionException, InterruptedException {
    throw new TargetErrorFunctionException(
        new TargetParsingException(((Key) skyKey).message, ((Key) skyKey).detailedExitCode),
        Transience.PERSISTENT);
  }

  private static class TargetErrorFunctionException extends SkyFunctionException {
    public TargetErrorFunctionException(TargetParsingException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
