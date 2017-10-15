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
package com.google.devtools.build.lib.analysis.extra;

import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides an action type -> set of extra actions to run map.
 */
@Immutable
public final class ExtraActionMapProvider implements TransitiveInfoProvider {
  private final ImmutableMultimap<String, ExtraActionSpec> extraActionMap;

  public ExtraActionMapProvider(Multimap<String, ExtraActionSpec> extraActionMap) {
    this.extraActionMap = ImmutableMultimap.copyOf(extraActionMap);
  }

  /**
   * Returns the extra action map.
   */
  public ImmutableMultimap<String, ExtraActionSpec> getExtraActionMap() {
    return extraActionMap;
  }
}
