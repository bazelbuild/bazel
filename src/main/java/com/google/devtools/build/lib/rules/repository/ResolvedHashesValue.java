// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;

/** The list of expected hashes of the directories */
public class ResolvedHashesValue implements SkyValue {
  @AutoCodec @AutoCodec.VisibleForSerialization
  static final SkyKey KEY = () -> SkyFunctions.RESOLVED_HASH_VALUES;

  private final Map<String, String> hashes;

  ResolvedHashesValue(Map<String, String> hashes) {
    this.hashes = hashes;
  }

  public Map<String, String> getHashes() {
    return hashes;
  }

  @Override
  public int hashCode() {
    return hashes.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof ResolvedHashesValue)) {
      return false;
    }
    return this.getHashes().equals(((ResolvedHashesValue) other).getHashes());
  }

  /** Returns the (singleton) {@link SkyKey} for {@link ResolvedHashesValue}s. */
  public static SkyKey key() {
    return KEY;
  }
}
