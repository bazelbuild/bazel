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
package com.google.devtools.build.lib.actions.util;

import com.google.devtools.build.lib.actions.cache.ActionCache;

import java.io.PrintStream;

/**
 * Utilities for tests that use the action cache.
 */
public class ActionCacheTestHelper {
  private ActionCacheTestHelper() {}

  /** A cache which does not remember anything.  Causes perpetual rebuilds! */
  public static final ActionCache AMNESIAC_CACHE =
    new ActionCache() {
      @Override
      public void put(String fingerprint, Entry entry) {}
      @Override
      public Entry get(String fingerprint) { return null; }
      @Override
      public void remove(String key) {}
      @Override
      public Entry createEntry(String key, boolean discoversInputs) {
        return new ActionCache.Entry(key, discoversInputs);
      }
      @Override
      public long save() { return -1; }
      @Override
      public void dump(PrintStream out) { }
    };
}
