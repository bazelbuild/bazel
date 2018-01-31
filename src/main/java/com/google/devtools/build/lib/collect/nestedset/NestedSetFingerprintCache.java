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

package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.actions.CommandLineItem;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Computes fingerprints for nested sets, reusing sub-computations from children. */
public class NestedSetFingerprintCache {
  private static final int DIGEST_SIZE = 16;
  private static final int EMPTY_SET_DIGEST = 104_395_303;

  /** Memoize the subresults. We have to have one cache per type of command item map function. */
  private Map<CommandLineItem.MapFn<?>, DigestMap> mapFnToFingerprints = createMap();

  public <T> void addNestedSetToFingerprint(Fingerprint fingerprint, NestedSet<T> nestedSet) {
    addNestedSetToFingerprint(CommandLineItem.MapFn.DEFAULT, fingerprint, nestedSet);
  }

  public <T> void addNestedSetToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, NestedSet<T> nestedSet) {
    // Only top-level nested sets can be empty, so we can bail here
    if (nestedSet.isEmpty()) {
      fingerprint.addInt(EMPTY_SET_DIGEST);
      return;
    }
    DigestMap digestMap =
        mapFnToFingerprints.computeIfAbsent(mapFn, k -> new DigestMap(DIGEST_SIZE, 1024));
    fingerprint.addInt(nestedSet.getOrder().ordinal());
    Object children = nestedSet.rawChildren();
    addToFingerprint(mapFn, fingerprint, digestMap, children);
  }

  public void clear() {
    mapFnToFingerprints = createMap();
  }

  @SuppressWarnings("unchecked")
  private <T> void addToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn,
      Fingerprint fingerprint,
      DigestMap digestMap,
      Object children) {
    if (children instanceof Object[]) {
      if (!digestMap.readDigest(children, fingerprint)) {
        Fingerprint childrenFingerprint = new Fingerprint();
        for (Object child : (Object[]) children) {
          addToFingerprint(mapFn, childrenFingerprint, digestMap, child);
        }
        digestMap.insertAndReadDigest(children, childrenFingerprint, fingerprint);
      }
    } else {
      addToFingerprint(mapFn, fingerprint, (T) children);
    }
  }

  @VisibleForTesting
  <T> void addToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, T object) {
    fingerprint.addString(mapFn.expandToCommandLine(object));
  }

  private static Map<CommandLineItem.MapFn<?>, DigestMap> createMap() {
    return new ConcurrentHashMap<>();
  }
}
