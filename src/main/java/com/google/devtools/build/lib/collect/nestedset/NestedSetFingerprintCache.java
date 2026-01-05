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
import com.google.devtools.build.lib.collect.nestedset.DigestDeduper.DigestReference;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** Computes fingerprints for nested sets, reusing sub-computations from children. */
public class NestedSetFingerprintCache {
  private static final int EMPTY_SET_DIGEST = 104_395_303;

  /** Memoize the subresults. We have to have one cache per type of command item map function. */
  private Map<CommandLineItem.MapFn<?>, DigestMap> mapFnToDigestMap = createMap();

  private final Set<Class<?>> seenMapFns = new HashSet<>();
  private final Multiset<Class<?>> seenParametrizedMapFns = HashMultiset.create();

  public <T> void addNestedSetToFingerprint(Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    addNestedSetToFingerprintExceptionless(CommandLineItem.MapFn.DEFAULT, fingerprint, nestedSet);
  }

  public <T> void addNestedSetToFingerprintExceptionless(
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

  @SuppressWarnings("EnumOrdinal") // ordinal not used across different binary versions
  public <T> void addNestedSetToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn, Fingerprint fingerprint, NestedSet<T> nestedSet)
      throws CommandLineExpansionException, InterruptedException {
    // Only top-level nested sets can be empty, so we can bail here
    if (nestedSet.isEmpty()) {
      fingerprint.addInt(EMPTY_SET_DIGEST);
      return;
    }
    DigestMap digestMap = mapFnToDigestMap.computeIfAbsent(mapFn, this::newDigestMap);
    fingerprint.addInt(nestedSet.getOrder().ordinal());
    Object children = nestedSet.getChildren();
    addToFingerprint(
        mapFn,
        children,
        digestMap,
        /* transitiveDigestDeduper= */ null,
        fingerprint,
        new DigestReference());
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

  public void clear() {
    mapFnToDigestMap = createMap();
    seenMapFns.clear();
    seenParametrizedMapFns.clear();
  }

  // safe cast of direct child to T after checking it's not a child array.
  @SuppressWarnings("unchecked")
  private <T> void addToFingerprint(
      CommandLineItem.MapFn<? super T> mapFn,
      Object rawChildren,
      DigestMap digestMap,
      // Deduplicator for any Object[] children in `rawChildren`. The top level `rawChildren`
      // instance has no siblings so this is null in that case. It's non-null on recursive calls.
      @Nullable DigestDeduper transitiveDigestDeduper,
      Fingerprint fingerprint,
      // A reusable buffer for digest references.
      DigestReference digestBuffer)
      throws CommandLineExpansionException, InterruptedException {
    if (!(rawChildren instanceof Object[] childArray)) {
      // It was an immediate child. These should already be deduplicated by the NestedSetBuilder.
      addToFingerprint(mapFn, fingerprint, (T) rawChildren);
      return;
    }

    if (digestMap.readDigest(childArray, digestBuffer)) {
      // Adds novel sets to the fingerprint and skips duplicates.
      if (transitiveDigestDeduper == null || transitiveDigestDeduper.add(digestBuffer)) {
        digestBuffer.addTo(fingerprint);
      }
      digestBuffer.clear();
      return;
    }

    Fingerprint childArrayFingerprinter = new Fingerprint();

    // `childArrayDeduper` is used to deduplicate transitive sets within the *same* direct child
    // array of a NestedSet. Note that Object[] children across different nodes of a NestedSet graph
    // cannot be deduplicated in this way because we are memoizing their fingerprints. For example,
    // let P be a parent Object[], U be an uncle Object[] and C be the child Object[]. Furthermore,
    // suppose that C is a duplicate of U. If we were to deduplicate C against U, the fingerprint of
    // P would change if P were reused in a different NestedSet without the presence of U, defeating
    // fingerprint memoization.
    var childArrayDeduper =
        new DigestDeduper(
            /* maxSize= */ childArray.length, /* digestLength= */ digestMap.getMaxDigestLength());

    for (Object child : childArray) {
      addToFingerprint(
          mapFn, child, digestMap, childArrayDeduper, childArrayFingerprinter, digestBuffer);
    }

    digestMap.insertAndReadDigest(childArray, childArrayFingerprinter, digestBuffer);
    if (transitiveDigestDeduper == null || transitiveDigestDeduper.add(digestBuffer)) {
      digestBuffer.addTo(fingerprint);
    }
    digestBuffer.clear();
  }

  // Non-static for testability.
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
