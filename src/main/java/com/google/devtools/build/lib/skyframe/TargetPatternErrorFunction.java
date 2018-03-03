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

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/**
 * SkyFunction that throws a {@link TargetParsingException} for target pattern that could not be
 * parsed. Must only be requested when a SkyFunction wishes to ignore the errors
 * in a target pattern in keep_going mode, but to shut down the build in nokeep_going mode.
 *
 * <p>This SkyFunction never returns a value, only throws a {@link TargetParsingException}, and
 * should never return null, since all of its dependencies should already be present.
 */
public class TargetPatternErrorFunction implements SkyFunction {
  // We pass in the error message, which isn't ideal. We could consider reparsing the original
  // pattern instead, but that requires more information.
  public static Key key(String errorMessage) {
    return Key.create(errorMessage);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<String> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(String arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(String arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_PATTERN_ERROR;
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws TargetErrorFunctionException, InterruptedException {
    String errorMessage = (String) skyKey.argument();
    throw new TargetErrorFunctionException(
        new TargetParsingException(errorMessage), Transience.PERSISTENT);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static class TargetErrorFunctionException extends SkyFunctionException {
    public TargetErrorFunctionException(
        TargetParsingException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
