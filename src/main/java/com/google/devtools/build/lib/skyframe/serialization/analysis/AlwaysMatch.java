// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.devtools.build.lib.skyframe.serialization.analysis.FileOpMatchResultTypes.FileOpMatchResult;
import com.google.devtools.build.lib.skyframe.serialization.analysis.NestedMatchResultTypes.NestedMatchResult;

/**
 * The delta matches any set of dependencies, meaning a <b>cache miss</b>.
 *
 * <p>This occurs when there is missing data, and thus no guarantee can be made about the cache
 * entry's validity.
 */
enum AlwaysMatch implements FileOpMatchResult, NestedMatchResult, MatchIndicator {
  ALWAYS_MATCH_RESULT;

  @Override
  public boolean isMatch() {
    return true;
  }

  @Override
  public final int version() {
    return VersionedChanges.ALWAYS_MATCH;
  }
}
