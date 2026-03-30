// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.platform;

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import java.util.Optional;

/**
 * A platform's {@link PlatformInfo} along with its parsed flags.
 *
 * @param parsedFlags Only present if the platform specifies flags.
 */
@AutoCodec
public record PlatformValue(PlatformInfo platformInfo, Optional<ParsedFlagsValue> parsedFlags)
    implements SkyValue {
  public PlatformValue {
    requireNonNull(platformInfo, "platformInfo");
    requireNonNull(parsedFlags, "parsedFlags");
  }

  static PlatformValue noFlags(PlatformInfo platformInfo) {
    return new PlatformValue(platformInfo, /* parsedFlags= */ Optional.empty());
  }

  static PlatformValue withFlags(PlatformInfo platformInfo, ParsedFlagsValue parsedFlags) {
    return new PlatformValue(platformInfo, Optional.of(parsedFlags));
  }

  public static Key key(Label platformLabel, ImmutableMap<String, Label> flagAliasMappings) {
    return Key.create(platformLabel, flagAliasMappings);
  }

  /** Key definition. */
  @AutoCodec
  public static final class Key implements SkyKey {
    private static final SkyKeyInterner<Key> interner = new SkyKeyInterner<>();

    private final Label label;
    private final ImmutableMap<String, Label> flagAliasMappings;
    private final int hashCode;

    private Key(Label label, ImmutableMap<String, Label> flagAliasMappings) {
      this.label = requireNonNull(label);
      this.flagAliasMappings = requireNonNull(flagAliasMappings);
      this.hashCode = Objects.hash(label, flagAliasMappings);
    }

    @AutoCodec.Instantiator
    static Key create(Label label, ImmutableMap<String, Label> flagAliasMappings) {
      return interner.intern(new Key(label, flagAliasMappings));
    }

    public Label label() {
      return label;
    }

    public ImmutableMap<String, Label> flagAliasMappings() {
      return flagAliasMappings;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PLATFORM;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Key key)) {
        return false;
      }
      return label.equals(key.label) && flagAliasMappings.equals(key.flagAliasMappings);
    }

    @Override
    public int hashCode() {
      return hashCode;
    }

    @Override
    public String toString() {
      return "Key[label=" + label + ", flagAliasMappings=" + flagAliasMappings + "]";
    }
  }
}
