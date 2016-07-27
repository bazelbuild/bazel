// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A marker provider for rules that are to be linked statically.
 *
 * Used in license checking.
 */
@Immutable
public class StaticallyLinkedMarkerProvider implements TransitiveInfoProvider {
  private final boolean isLinkedStatically;

  public StaticallyLinkedMarkerProvider(boolean isLinkedStatically) {
    this.isLinkedStatically = isLinkedStatically;
  }

  public boolean isLinkedStatically() {
    return isLinkedStatically;
  }
}
