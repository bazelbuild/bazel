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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.util.Fingerprint;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NestedSetFingerprintCache}. */
@RunWith(JUnit4.class)
public class NestedSetFingerprintCacheTest {

  private class TestNestedSetFingerprintCache extends NestedSetFingerprintCache {
    private final Multiset<Object> fingerprinted = HashMultiset.create();

    @Override
    <T> void addToFingerprint(Fingerprint fingerprint, T object) {
      super.addToFingerprint(fingerprint, object);
      fingerprinted.add(object);
    }
  }

  private TestNestedSetFingerprintCache cache;

  @Before
  public void setup() {
    cache = new TestNestedSetFingerprintCache();
  }

  @Test
  public void testBasic() {
    NestedSet<String> nestedSet = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();

    // This test does reimplement the inner algorithm of the cache, but serves
    // as a simple check that the basic operations do something sensible
    Fingerprint fingerprint = new Fingerprint();
    fingerprint.addInt(nestedSet.getOrder().ordinal());
    Fingerprint subFingerprint = new Fingerprint();
    subFingerprint.addString("a");
    subFingerprint.addString("b");
    fingerprint.addBytes(subFingerprint.digestAndReset());
    String controlDigest = fingerprint.hexDigestAndReset();

    Fingerprint nestedSetFingerprint = new Fingerprint();
    cache.addNestedSetToFingerprint(nestedSetFingerprint, nestedSet);
    String nestedSetDigest = nestedSetFingerprint.hexDigestAndReset();

    assertThat(controlDigest).isEqualTo(nestedSetDigest);
  }

  @Test
  public void testOnlyFingerprintedOncePerString() {
    // Leaving leaf nodes with a single item will defeat this check
    // The nested set builder will effectively inline single-item objects into their parent,
    // meaning they will get hashed multiple times.
    NestedSet<String> a = NestedSetBuilder.<String>stableOrder().add("a0").add("a1").build();
    NestedSet<String> b = NestedSetBuilder.<String>stableOrder().add("b0").add("b1").build();
    NestedSet<String> c =
        NestedSetBuilder.<String>stableOrder().add("c").addTransitive(a).addTransitive(b).build();
    NestedSet<String> d =
        NestedSetBuilder.<String>stableOrder().add("d").addTransitive(a).addTransitive(b).build();
    NestedSet<String> e =
        NestedSetBuilder.<String>stableOrder().add("e").addTransitive(c).addTransitive(d).build();
    cache.addNestedSetToFingerprint(new Fingerprint(), e);
    assertThat(cache.fingerprinted.elementSet())
        .containsExactly("a0", "a1", "b0", "b1", "c", "d", "e");
    for (Multiset.Entry<Object> entry : cache.fingerprinted.entrySet()) {
      assertThat(entry.getCount()).isEqualTo(1);
    }
  }
}
