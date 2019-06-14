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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.base.Objects;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import com.google.devtools.build.lib.actions.CommandLineItem;
import com.google.devtools.build.lib.actions.CommandLineItem.CapturingMapFn;
import com.google.devtools.build.lib.actions.CommandLineItem.MapFn;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.function.Consumer;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NestedSetFingerprintCache}. */
@RunWith(JUnit4.class)
public class NestedSetFingerprintCacheTest {

  private class TestNestedSetFingerprintCache extends NestedSetFingerprintCache {
    private Multiset<Object> fingerprinted = HashMultiset.create();

    @Override
    <T> void addToFingerprint(MapFn<? super T> mapFn, Fingerprint fingerprint, T object) {
      super.addToFingerprint(mapFn, fingerprint, object);
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

  @Test
  public void testMapFn() {
    // Make sure that the map function assigns completely different key spaces
    NestedSet<String> a = NestedSetBuilder.<String>stableOrder().add("a0").add("a1").build();

    Fingerprint defaultMapFnFingerprint = new Fingerprint();
    cache.addNestedSetToFingerprint(defaultMapFnFingerprint, a);
    Fingerprint explicitDefaultMapFnFingerprint = new Fingerprint();
    cache.addNestedSetToFingerprint(
        CommandLineItem.MapFn.DEFAULT, explicitDefaultMapFnFingerprint, a);
    Fingerprint mappedFingerprint = new Fingerprint();
    cache.addNestedSetToFingerprint((s, args) -> args.accept(s + "_mapped"), mappedFingerprint, a);

    String defaultMapFnDigest = defaultMapFnFingerprint.hexDigestAndReset();
    String explicitDefaultMapFnDigest = explicitDefaultMapFnFingerprint.hexDigestAndReset();
    String mappedDigest = mappedFingerprint.hexDigestAndReset();
    assertThat(defaultMapFnDigest).isEqualTo(explicitDefaultMapFnDigest);
    assertThat(mappedDigest).isNotEqualTo(defaultMapFnDigest);

    assertThat(cache.fingerprinted.elementSet()).containsExactly("a0", "a1");
    for (Multiset.Entry<Object> entry : cache.fingerprinted.entrySet()) {
      assertThat(entry.getCount()).isEqualTo(2);
    }
  }

  @Test
  public void testMultipleInstancesOfMapFnThrows() {
    NestedSet<String> nestedSet =
        NestedSetBuilder.<String>stableOrder().add("a0").add("a1").build();

    // Make sure a normal method reference doesn't get blacklisted.
    for (int i = 0; i < 2; ++i) {
      cache.addNestedSetToFingerprint(
          NestedSetFingerprintCacheTest::simpleExpand, new Fingerprint(), nestedSet);
    }

    // Try again to make sure Java synthesizes a new class for a second method reference.
    for (int i = 0; i < 2; ++i) {
      cache.addNestedSetToFingerprint(
          NestedSetFingerprintCacheTest::simpleExpand2, new Fingerprint(), nestedSet);
    }

    // Make sure a non-capturing lambda doesn't get blacklisted
    for (int i = 0; i < 2; ++i) {
      cache.addNestedSetToFingerprint(
          (s, args) -> args.accept(s + "_mapped"), new Fingerprint(), nestedSet);
    }

    // Make sure a CapturingMapFn doesn't get blacklisted
    for (int i = 0; i < 2; ++i) {
      cache.addNestedSetToFingerprint(
          (CapturingMapFn<String>) (s, args) -> args.accept(s + 1), new Fingerprint(), nestedSet);
    }

    // Make sure a ParametrizedMapFn doesn't get blacklisted until it exceeds its instance count
    cache.addNestedSetToFingerprint(new IntParametrizedMapFn(1), new Fingerprint(), nestedSet);
    cache.addNestedSetToFingerprint(new IntParametrizedMapFn(2), new Fingerprint(), nestedSet);
    assertThrows(
        IllegalArgumentException.class,
        () ->
            cache.addNestedSetToFingerprint(
                new IntParametrizedMapFn(3), new Fingerprint(), nestedSet));

    // Make sure a capturing method reference gets blacklisted. The for loop causes the variable i
    // to be captured, so that str::expand becomes a capturing lambda, not a plain method reference.
    // This test case ensures that the captured lambda cannot be used twice.
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          for (int i = 0; i < 2; ++i) {
            StringJoiner str = new StringJoiner("hello");
            cache.addNestedSetToFingerprint(str::expand, new Fingerprint(), nestedSet);
          }
        });

    // Do make sure that a capturing lambda gets blacklisted. The loop exists for the same reason as
    // the above case.
    assertThrows(
        IllegalArgumentException.class,
        () -> {
          for (int i = 0; i < 2; ++i) {
            final int capturedVariable = i;
            cache.addNestedSetToFingerprint(
                (s, args) -> args.accept(s + capturedVariable), new Fingerprint(), nestedSet);
          }
        });
  }

  private static class IntParametrizedMapFn extends CommandLineItem.ParametrizedMapFn<String> {
    private final int i;

    private IntParametrizedMapFn(int i) {
      this.i = i;
    }

    @Override
    public void expandToCommandLine(String object, Consumer<String> args) {
      args.accept(object + i);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      IntParametrizedMapFn that = (IntParametrizedMapFn) o;
      return i == that.i;
    }

    @Override
    public int maxInstancesAllowed() {
      return 2;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(i);
    }
  }

  private static class StringJoiner {
    private final String str;

    private StringJoiner(String str) {
      this.str = str;
    }

    private void expand(String other, Consumer<String> args) {
      args.accept(str + other);
    }
  }

  private static void simpleExpand(String o, Consumer<String> args) {
    args.accept(o + "_mapped");
  }

  private static void simpleExpand2(String o, Consumer<String> args) {
    args.accept(o + "_mapped2");
  }
}
