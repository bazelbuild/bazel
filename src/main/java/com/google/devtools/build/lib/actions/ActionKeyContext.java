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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetFingerprintCache;
import com.google.devtools.build.lib.util.Fingerprint;

/** Contains state that aids in action key computation via {@link AbstractAction#computeKey}. */
public class ActionKeyContext {

  private final NestedSetFingerprintCache nestedSetFingerprintCache =
      new NestedSetFingerprintCache();

  public <T> void addNestedSetToFingerprint(Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    nestedSetFingerprintCache.addNestedSetToFingerprint(fingerprint, nestedSet);
  }

  public <T> void addNestedSetToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    nestedSetFingerprintCache.addNestedSetToFingerprint(mapFn, fingerprint, nestedSet);
  }

  public <T> void addNestedSetToFingerprint(
      CommandLineItem.ExceptionlessMapFn<? super T> mapFn,
      Fingerprint fingerprint,
      NestedSet<T> nestedSet) {
    nestedSetFingerprintCache.addNestedSetToFingerprint(mapFn, fingerprint, nestedSet);
  }

  public static <T> String describeNestedSetFingerprint(
      CommandLineItem.ExceptionlessMapFn<? super T> mapFn, NestedSet<T> nestedSet) {
    return NestedSetFingerprintCache.describedNestedSetFingerprint(mapFn, nestedSet);
  }

  public void clear() {
    nestedSetFingerprintCache.clear();
  }
}
