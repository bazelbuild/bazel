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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.Serializable;

/** Represents a list of c++ tool flags. */
@AutoCodec
@Immutable
public class FlagList implements Serializable {
  private final ImmutableList<String> prefixFlags;
  private final ImmutableList<String> suffixFlags;

  @AutoCodec.Instantiator
  FlagList(
      ImmutableList<String> prefixFlags,
      ImmutableList<String> suffixFlags) {
    this.prefixFlags = prefixFlags;
    this.suffixFlags = suffixFlags;
  }

  @VisibleForTesting
  ImmutableList<String> evaluate() {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    result.addAll(prefixFlags);
    result.addAll(suffixFlags);
    return result.build();
  }
}
