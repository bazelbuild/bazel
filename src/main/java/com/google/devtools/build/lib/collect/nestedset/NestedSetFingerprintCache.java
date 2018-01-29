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
  private static final byte[] EMPTY_SET_BYTES = new byte[] {};

  /** Memoize the subresults. We have to have one cache per type of command item map function. */
  private Map<CommandLineItem.MapFn<?>, Map<Object, byte[]>> mapFnToFingerprints = createMap();

  public <T> void addNestedSetToFingerprint(Fingerprint fingerprint, NestedSet<T> nestedSet) {
    addNestedSetToFingerprint(CommandLineItem.MapFn.DEFAULT, fingerprint, nestedSet);
  }

  public <T> void addNestedSetToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, NestedSet<T> nestedSet) {
    Map<Object, byte[]> fingerprints =
        mapFnToFingerprints.computeIfAbsent(mapFn, k -> new ConcurrentHashMap<>());
    fingerprint.addInt(nestedSet.getOrder().ordinal());
    Object children = nestedSet.rawChildren();
    byte[] bytes = getBytes(mapFn, fingerprints, children);
    fingerprint.addBytes(bytes);
  }

  public void clear() {
    mapFnToFingerprints = createMap();
  }

  @SuppressWarnings("unchecked")
  private <T> byte[] getBytes(
      CommandLineItem.MapFn<? super T> mapFn, Map<Object, byte[]> fingerprints, Object children) {
    byte[] bytes = fingerprints.get(children);
    if (bytes == null) {
      if (children instanceof Object[]) {
        Fingerprint fingerprint = new Fingerprint();
        for (Object child : (Object[]) children) {
          if (child instanceof Object[]) {
            fingerprint.addBytes(getBytes(mapFn, fingerprints, child));
          } else {
            addToFingerprint(mapFn, fingerprint, (T) child);
          }
        }
        bytes = fingerprint.digestAndReset();

        // There is no point memoizing anything except the multi-item case,
        // since the single-item case gets inlined into its parents anyway,
        // and the empty set is a singleton
        fingerprints.put(children, bytes);
      } else if (children != NestedSet.EMPTY_CHILDREN) {
        // Single item
        Fingerprint fingerprint = new Fingerprint();
        addToFingerprint(mapFn, fingerprint, (T) children);
        bytes = fingerprint.digestAndReset();
      } else {
        // Empty nested set
        bytes = EMPTY_SET_BYTES;
      }
    }
    return bytes;
  }

  @VisibleForTesting
  <T> void addToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, T object) {
    fingerprint.addString(mapFn.expandToCommandLine(object));
  }

  private static Map<CommandLineItem.MapFn<?>, Map<Object, byte[]>> createMap() {
    return new ConcurrentHashMap<>();
  }
}
