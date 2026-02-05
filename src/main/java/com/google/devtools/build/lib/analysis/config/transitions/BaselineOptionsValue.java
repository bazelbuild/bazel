// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config.transitions;

import static java.util.Objects.requireNonNull;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CheckReturnValue;
import javax.annotation.Nullable;

/**
 * This contains the baseline options to compare against when constructing output paths.
 *
 * <p>When constructing the output mnemonic as part of making a {@link BuildConfigurationValue} and
 * the selected naming scheme is to diff against a baseline, this function returns the baseline to
 * use for that comparison. Differences in options between the given option and this baseline will
 * then be used to append a deconflicting ST-hash to the output mnemonic.
 *
 * <p>The afterExecTransition option in the key will apply the exec transition to the usual
 * baseline. It is expected that this is set whenever the given options have isExec set (and thus an
 * exec transition has already been applied to those options). The expectation here is that, as the
 * exec transition particularly sets many options, comparing against a post-exec baseline will yield
 * fewer differences. Note that some indicator must be added to the mnemonic (e.g. -exec-) in order
 * to deconflict for similar options where isExec is not set.
 *
 * <p>Similarly, the trimTestOptions option in the key will apply test trimming to the usual
 * baseline to reduce the number of differences for non-test targets.
 */
@CheckReturnValue
@Immutable
@ThreadSafe
@AutoCodec
public record BaselineOptionsValue(BuildOptions toOptions) implements SkyValue {
  public BaselineOptionsValue {
    requireNonNull(toOptions, "toOptions");
  }

  public static BaselineOptionsValue create(BuildOptions toOptions) {
    return new BaselineOptionsValue(toOptions);
  }

  public static Key key(
      boolean afterExecTransition, boolean trimTestOptions, @Nullable Label newPlatform) {
    return Key.create(afterExecTransition, trimTestOptions, newPlatform);
  }

  /** {@link SkyKey} implementation used for {@link BaselineOptionsValue}. */
  @CheckReturnValue
  @Immutable
  @ThreadSafe
  @AutoCodec
  public record Key(
      boolean afterExecTransition, boolean trimTestOptions, @Nullable Label newPlatform)
      implements SkyKey {

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BASELINE_OPTIONS;
    }

    @Override
    public String toString() {
      return "BaselineOptionsValue.Key{afterExecTransition=%s, trimTestOptions=%s, newPlatform=%s}"
          .formatted(afterExecTransition(), trimTestOptions(), newPlatform());
    }

    static Key create(
        boolean afterExecTransition, boolean trimTestOptions, @Nullable Label newPlatform) {
      return new Key(afterExecTransition, trimTestOptions, newPlatform);
    }
  }
}
