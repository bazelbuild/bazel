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
import javax.annotation.Nullable;

/**
 * A wrapper class for an {@code ImmutableSortedSet<Class<? extends BuildConfiguration.Fragment>>}
 * object. Interning these objects allows us to do cheap reference equality checks when these sets
 * are in frequently used keys. For good measure, we also compute a fingerprint.
 */
@AutoCodec
public class FragmentClassSet {
  private final ImmutableSortedSet<Class<? extends Fragment>> fragments;

  // Lazily initialized.
  @Nullable private volatile byte[] fingerprint;
  private volatile int hashCode;

  private FragmentClassSet(ImmutableSortedSet<Class<? extends Fragment>> fragments) {
    this.fragments = fragments;
  }

  private static final Interner<FragmentClassSet> interner = BlazeInterners.newWeakInterner();

  @AutoCodec.Instantiator
  public static FragmentClassSet of(ImmutableSortedSet<Class<? extends Fragment>> fragments) {
    return interner.intern(new FragmentClassSet(fragments));
  }

  public ImmutableSortedSet<Class<? extends Fragment>> fragmentClasses() {
    return fragments;
  }

  /**
   * Lazily initialize {@link #fingerprint} and {@link #hashCode}. Keeps computation off critical
   * path of build, while still avoiding expensive computation for equality and hash code each time.
   *
   * <p>We check for nullity of {@link #fingerprint} to see if this method has already been called.
   * Using {@link #hashCode} after this method is called is safe because it is set here before
   * {@link #fingerprint} is set, so if {@link #fingerprint} is non-null then {@link #hashCode} is
   * definitely set.
   */
  private void maybeInitializeFingerprintAndHashCode() {
    if (fingerprint != null) {
      return;
    }
    synchronized (this) {
      if (fingerprint != null) {
        return;
      }
      Fingerprint fingerprint = new Fingerprint();
      for (Class<? extends Fragment> fragment : fragments) {
        fingerprint.addString(fragment.getName());
      }
      byte[] computedFingerprint = fingerprint.digestAndReset();
      hashCode = Arrays.hashCode(computedFingerprint);
      this.fingerprint = computedFingerprint;
    }
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (!(other instanceof FragmentClassSet)) {
      return false;
    } else {
      maybeInitializeFingerprintAndHashCode();
      FragmentClassSet that = (FragmentClassSet) other;
      that.maybeInitializeFingerprintAndHashCode();
      return Arrays.equals(this.fingerprint, that.fingerprint);
    }
  }

  @Override
  public int hashCode() {
    maybeInitializeFingerprintAndHashCode();
    return hashCode;
  }

  @Override
  public String toString() {
    return String.format(
        "FragmentClassSet[%s]", fragments.stream().map(Class::getName).collect(joining(",")));
  }
}
