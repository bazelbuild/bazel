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

package com.google.devtools.build.lib.analysis.config;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Arrays;

/**
 * A wrapper class for an {@code ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>>}
 * object. Interning these objects allows us to do cheap reference equality checks when these sets
 * are in frequently used keys. For good measure, we also compute a fingerprint.
 */
@AutoCodec
public final class FragmentClassSet {

  private static final Interner<FragmentClassSet> interner = BlazeInterners.newWeakInterner();

  @AutoCodec.Instantiator
  public static FragmentClassSet of(ImmutableSortedSet<Class<? extends Fragment>> fragments) {
    byte[] fingerprint = computeFingerprint(fragments);
    return interner.intern(
        new FragmentClassSet(fragments, fingerprint, Arrays.hashCode(fingerprint)));
  }

  private static byte[] computeFingerprint(
      ImmutableSortedSet<Class<? extends Fragment>> fragments) {
    Fingerprint fingerprint = new Fingerprint();
    for (Class<?> fragment : fragments) {
      fingerprint.addString(fragment.getName());
    }
    return fingerprint.digestAndReset();
  }

  private final ImmutableSortedSet<Class<? extends Fragment>> fragments;
  private final byte[] fingerprint;
  private final int hashCode;

  private FragmentClassSet(
      ImmutableSortedSet<Class<? extends Fragment>> fragments, byte[] fingerprint, int hashCode) {
    this.fragments = fragments;
    this.fingerprint = fingerprint;
    this.hashCode = hashCode;
  }

  public ImmutableSortedSet<Class<? extends Fragment>> fragmentClasses() {
    return fragments;
  }

  @Override
  @SuppressWarnings("ReferenceEquality") // Fast-path check of the underlying fragments set.
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof FragmentClassSet)) {
      return false;
    }
    FragmentClassSet that = (FragmentClassSet) other;
    return fragments == that.fragments || Arrays.equals(fingerprint, that.fingerprint);
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public String toString() {
    return String.format(
        "FragmentClassSet[%s]", fragments.stream().map(Class::getName).collect(joining(",")));
  }
}
