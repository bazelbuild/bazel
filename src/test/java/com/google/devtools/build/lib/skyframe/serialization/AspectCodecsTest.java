// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.StarlarkAspectClass;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for serialization of aspect-related classes. */
@RunWith(JUnit4.class)
public final class AspectCodecsTest {

  @Test
  public void testStarlarkAspectClassMemoization() throws Exception {
    // We create two wholly different object graphs for the two objects to be serialized
    BzlLoadValue.Key key1 =
        BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl"));
    BzlLoadValue.Key key2 =
        BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl"));

    StarlarkAspectClass class1 = new StarlarkAspectClass(key1, "my_aspect");
    StarlarkAspectClass class2 = new StarlarkAspectClass(key2, "my_aspect");

    assertMemoized(class1, class2);
  }

  @Test
  public void testAspectDescriptorMemoization() throws Exception {
    // We create two wholly different object graphs for the two objects to be serialized
    BzlLoadValue.Key key1 =
        BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl"));
    BzlLoadValue.Key key2 =
        BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl"));

    StarlarkAspectClass aspectClass1 = new StarlarkAspectClass(key1, "my_aspect");
    StarlarkAspectClass aspectClass2 = new StarlarkAspectClass(key2, "my_aspect");

    AspectDescriptor d1 =
        AspectDescriptor.createUninternedForTesting(aspectClass1, AspectParameters.EMPTY);
    AspectDescriptor d2 =
        AspectDescriptor.createUninternedForTesting(aspectClass2, AspectParameters.EMPTY);

    assertMemoized(d1, d2);
  }

  @Test
  public void testAspectMemoization() throws Exception {
    // We create two mostly different object graphs for the two objects to be serialized
    BzlLoadValue.Key key1 =
        BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl"));
    BzlLoadValue.Key key2 =
        BzlLoadValue.keyForBuild(Label.parseCanonicalUnchecked("//foo:bar.bzl"));

    StarlarkAspectClass aspectClass1 = new StarlarkAspectClass(key1, "my_aspect");
    StarlarkAspectClass aspectClass2 = new StarlarkAspectClass(key2, "my_aspect");

    // AspectDefinition doesn't override .equals() so we only create one instance. It is memoized
    // but uses a non-weak interner so it should be fine.
    AspectDefinition definition = new AspectDefinition.Builder(aspectClass1).build();

    AspectDescriptor descriptor1 = AspectDescriptor.of(aspectClass1, AspectParameters.EMPTY);
    AspectDescriptor descriptor2 = AspectDescriptor.of(aspectClass2, AspectParameters.EMPTY);

    Aspect a1 = Aspect.createUninternedForTesting(descriptor1, definition);
    Aspect a2 = Aspect.createUninternedForTesting(descriptor2, definition);

    assertMemoized(a1, a2);
  }

  private static <T> void assertMemoized(T obj1, T obj2) throws Exception {
    assertThat(obj1).isNotSameInstanceAs(obj2);
    assertThat(obj1).isEqualTo(obj2);

    new SerializationTester(ImmutableList.of(ImmutableList.of(obj1, obj2)))
        .makeMemoizing()
        .setVerificationFunction(
            (original, deserialized) -> {
              @SuppressWarnings("unchecked")
              ImmutableList<T> deserializedList = (ImmutableList<T>) deserialized;
              assertThat(deserializedList).hasSize(2);
              assertThat(deserializedList.get(0)).isSameInstanceAs(deserializedList.get(1));
            })
        .runTests();
  }
}
