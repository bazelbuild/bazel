// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import com.google.devtools.common.options.BooleanStyleOption;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Locale;
import java.util.Set;

/** Configuration for slim profiling. */
public final class SlimProfileConfiguration {
  private static final SlimProfileConfiguration DISABLED = new SlimProfileConfiguration(false, -1);
  private static final SlimProfileConfiguration ALWAYS = new SlimProfileConfiguration(true, -1);

  private final boolean enabled;
  private final long sizeLimit; // in bytes. <= 0 means no limit (always slim if enabled)

  private SlimProfileConfiguration(boolean enabled, long sizeLimit) {
    this.enabled = enabled;
    this.sizeLimit = sizeLimit;
  }

  public static SlimProfileConfiguration disabled() {
    return DISABLED;
  }

  public static SlimProfileConfiguration always() {
    return ALWAYS;
  }

  public static SlimProfileConfiguration afterSize(long sizeLimit) {
    return new SlimProfileConfiguration(true, sizeLimit);
  }

  public boolean isEnabled() {
    return enabled;
  }

  public long getSizeLimit() {
    return sizeLimit;
  }

  public boolean hasSizeLimit() {
    return enabled && sizeLimit > 0;
  }

  @Override
  public String toString() {
    if (!enabled) {
      return "disabled";
    }
    if (sizeLimit <= 0) {
      return "enabled (always)";
    }
    return "enabled (after " + sizeLimit + " bytes)";
  }

  /** Converter for {@link SlimProfileConfiguration}. */
  public static final class SlimProfileConverter
      extends Converter.Contextless<SlimProfileConfiguration> implements BooleanStyleOption {
    private static final Set<String> ENABLED_REPS = Set.of("true", "1", "yes", "t", "y");
    private static final Set<String> DISABLED_REPS = Set.of("false", "0", "no", "f", "n");

    @Override
    public SlimProfileConfiguration convert(String input) throws OptionsParsingException {
      if (input == null) {
        return SlimProfileConfiguration.disabled();
      }
      String lowerInput = input.toLowerCase(Locale.ENGLISH);
      if (ENABLED_REPS.contains(lowerInput)) {
        return SlimProfileConfiguration.always();
      }
      if (DISABLED_REPS.contains(lowerInput)) {
        return SlimProfileConfiguration.disabled();
      }

      try {
        long value = new Converters.ByteSizeConverter().convert(input);
        return SlimProfileConfiguration.afterSize(value);
      } catch (OptionsParsingException e) {
        throw new OptionsParsingException(
            "Expected a boolean or a size in bytes (e.g. 5M), but got: " + input, e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a boolean or a size in bytes (e.g. 5M)";
    }
  }
}
