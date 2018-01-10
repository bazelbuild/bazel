// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.io.Serializable;
import java.util.List;

/** Represents a list of c++ tool flags. */
@AutoCodec
@Immutable
public class FlagList implements Serializable {
  public static final ObjectCodec<FlagList> CODEC = new FlagList_AutoCodec();

  /** Represents an optional flag that can be toggled using the package features mechanism. */
  @AutoCodec
  @Immutable
  static class OptionalFlag implements Serializable {
    public static final ObjectCodec<OptionalFlag> CODEC = new FlagList_OptionalFlag_AutoCodec();

    private final String name;
    private final ImmutableList<String> flags;

    @AutoCodec.Constructor
    OptionalFlag(String name, ImmutableList<String> flags) {
      this.name = name;
      this.flags = flags;
    }

    private ImmutableList<String> getFlags() {
      return flags;
    }

    private String getName() {
      return name;
    }
  }

  private final ImmutableList<String> prefixFlags;
  private final ImmutableList<OptionalFlag> optionalFlags;
  private final ImmutableList<String> suffixFlags;

  @AutoCodec.Constructor
  FlagList(
      ImmutableList<String> prefixFlags,
      ImmutableList<OptionalFlag> optionalFlags,
      ImmutableList<String> suffixFlags) {
    this.prefixFlags = prefixFlags;
    this.optionalFlags = optionalFlags;
    this.suffixFlags = suffixFlags;
  }

  static ImmutableList<OptionalFlag> convertOptionalOptions(
      List<CToolchain.OptionalFlag> optionalFlagList) {
    ImmutableList.Builder<OptionalFlag> result = ImmutableList.builder();

    for (CToolchain.OptionalFlag crosstoolOptionalFlag : optionalFlagList) {
      String name = crosstoolOptionalFlag.getDefaultSettingName();
      result.add(new OptionalFlag(name, ImmutableList.copyOf(crosstoolOptionalFlag.getFlagList())));
    }

    return result.build();
  }

  @VisibleForTesting
  ImmutableList<String> evaluate(Iterable<String> features) {
    ImmutableSet<String> featureSet = ImmutableSet.copyOf(features);
    ImmutableList.Builder<String> result = ImmutableList.builder();
    result.addAll(prefixFlags);
    for (OptionalFlag optionalFlag : optionalFlags) {
      // The flag is added if the default is true and the flag is not specified,
      // or if the default is false and the flag is specified.
      if (featureSet.contains(optionalFlag.getName())) {
        result.addAll(optionalFlag.getFlags());
      }
    }

    result.addAll(suffixFlags);
    return result.build();
  }
}
