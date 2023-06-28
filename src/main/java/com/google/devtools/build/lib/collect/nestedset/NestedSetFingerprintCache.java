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
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.CommandLineItem.MapFn;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** Computes fingerprints for nested sets, reusing sub-computations from children. */
public class NestedSetFingerprintCache {
  private static final int EMPTY_SET_DIGEST = 104_395_303;

  /** Memoize the subresults. We have to have one cache per type of command item map function. */
  private Map<CommandLineItem.MapFn<?>, DigestMap> mapFnToDigestMap = createMap();

  private final Set<Class<?>> seenMapFns = new HashSet<>();
  private final Multiset<Class<?>> seenParametrizedMapFns = HashMultiset.create();

  public <T> void addNestedSetToFingerprint(Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    addNestedSetToFingerprint(CommandLineItem.MapFn.DEFAULT, fingerprint, nestedSet);
  }

  public <T> void addNestedSetToFingerprint(
      CommandLineItem.ExceptionlessMapFn<? super T> mapFn,
      Fingerprint fingerprint,
      NestedSet<T> nestedSet) {
    try {
      addNestedSetToFingerprint((CommandLineItem.MapFn<? super T>) mapFn, fingerprint, nestedSet);
    } catch (CommandLineExpansionException | InterruptedException e) {
      // addNestedSetToFingerprint only throws these exceptions if mapFn does.
      throw new IllegalStateException(e);
    }
  }

  public <T> void addNestedSetToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    if (mapFn instanceof CommandLineItem.CapturingMapFn) {
      addNestedSetToFingerprintSlow(mapFn, fingerprint, nestedSet);
      return;
    }
    // Only top-level nested sets can be empty, so we can bail here
    if (nestedSet.isEmpty()) {
      fingerprint.addInt(EMPTY_SET_DIGEST);
      return;
    }
    DigestMap digestMap = mapFnToDigestMap.computeIfAbsent(mapFn, this::newDigestMap);
    fingerprint.addInt(nestedSet.getOrder().ordinal());
    Object children = nestedSet.getChildren();
    addToFingerprint(mapFn, fingerprint, digestMap, children);
  }

  public static <T> String describedNestedSetFingerprint(
      CommandLineItem.ExceptionlessMapFn<? super T> mapFn, NestedSet<T> nestedSet) {
    if (nestedSet.isEmpty()) {
      return "<empty>";
    }
    StringBuilder sb = new StringBuilder();
    sb.append("order: ")
        .append(nestedSet.getOrder())
        .append(
            " (fingerprinting considers internal"
                + " nested set structure, which is not reflected in values reported below)\n");
    ImmutableList<T> list = nestedSet.toList();
    sb.append("size: ").append(list.size()).append('\n');
    for (T item : list) {
      sb.append("  ");
      mapFn.expandToCommandLine(item, s -> sb.append(s).append(", "));
      sb.append('\n');
    }
    return sb.toString();
  }

  private <T> void addNestedSetToFingerprintSlow(
      MapFn<? super T> mapFn, Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    for (T object : nestedSet.toList()) {
      addToFingerprint(mapFn, fingerprint, object);
    }
  }

  public void clear() {
    mapFnToDigestMap = createMap();
    seenMapFns.clear();
    seenParametrizedMapFns.clear();
  }

  @SuppressWarnings("unchecked")
  private <T> void addToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn,
      Fingerprint fingerprint,
      DigestMap digestMap,
      Object children)
      throws CommandLineExpansionException, InterruptedException {
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
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, T object)
      throws CommandLineExpansionException, InterruptedException {
    mapFn.expandToCommandLine(object, fingerprint::addString);
  }

  private static Map<CommandLineItem.MapFn<?>, DigestMap> createMap() {
    return new ConcurrentHashMap<>();
  }

  private DigestMap newDigestMap(CommandLineItem.MapFn<?> mapFn) {
    Class<?> mapFnClass = mapFn.getClass();
    if (mapFn instanceof CommandLineItem.ParametrizedMapFn) {
      int occurrences = seenParametrizedMapFns.add(mapFnClass, 1) + 1;
      if (occurrences > ((CommandLineItem.ParametrizedMapFn) mapFn).maxInstancesAllowed()) {
        throw new IllegalArgumentException(
            String.format(
                "Too many instances of CommandLineItem.ParametrizedMapFn '%s' detected. "
                    + "Please construct fewer instances.",
                mapFnClass.getName()));
      }
    } else {
      if (!seenMapFns.add(mapFnClass)) {
        throw new IllegalArgumentException(
            String.format(
                "Illegal mapFn implementation: '%s'. "
                    + "CommandLineItem.MapFn instances must be singletons."
                    + "Please see CommandLineItem.ParametrizedMapFn for an alternative.",
                mapFnClass.getName()));
      }
    }
    // TODO(b/112460990): Use the value from DigestHashFunction.getDefault(), but check for
    // contention.
    return new DigestMap(DigestHashFunction.SHA256, 1024);
  }
}
