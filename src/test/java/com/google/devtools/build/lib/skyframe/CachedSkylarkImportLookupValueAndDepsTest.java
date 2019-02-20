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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link CachedSkylarkImportLookupValueAndDeps}. */
@RunWith(JUnit4.class)
public class CachedSkylarkImportLookupValueAndDepsTest {
  @Test
  public void testDepsAreNotVisitedMultipleTimesForDiamondDependencies() throws Exception {
    // Graph structure of SkylarkImportLookupValues:
    //
    //     p
    //   /  \
    //  c1  c2
    //   \  /
    //    gc

    SkylarkImportLookupValue dummyValue = Mockito.mock(SkylarkImportLookupValue.class);

    SkyKey gcKey1 = createKey("gc key1");
    SkyKey gcKey2 = createKey("gc key2");
    SkyKey gcKey3 = createKey("gc key3");
    CachedSkylarkImportLookupValueAndDeps gc =
        CachedSkylarkImportLookupValueAndDeps.newBuilder()
            .addDep(gcKey1)
            .addDeps(ImmutableList.of(gcKey2, gcKey3))
            .setValue(dummyValue)
            .build();

    SkyKey c1Key1 = createKey("c1 key1");
    CachedSkylarkImportLookupValueAndDeps c1 =
        CachedSkylarkImportLookupValueAndDeps.newBuilder()
            .addDep(c1Key1)
            .addTransitiveDeps(gc)
            .setValue(dummyValue)
            .build();

    SkyKey c2Key1 = createKey("c2 key1");
    SkyKey c2Key2 = createKey("c2 key2");
    CachedSkylarkImportLookupValueAndDeps c2 =
        CachedSkylarkImportLookupValueAndDeps.newBuilder()
            .addDeps(ImmutableList.of(c2Key1, c2Key2))
            .addTransitiveDeps(gc)
            .setValue(dummyValue)
            .build();

    SkyKey pKey1 = createKey("p key1");
    CachedSkylarkImportLookupValueAndDeps p =
        CachedSkylarkImportLookupValueAndDeps.newBuilder()
            .addDep(pKey1)
            .addTransitiveDeps(c1)
            .addTransitiveDeps(c2)
            .setValue(dummyValue)
            .build();

    List<Iterable<SkyKey>> registeredDeps = new ArrayList<>();
    p.traverse(registeredDeps::add, /*visitedGlobalDeps=*/ new HashSet<>());

    assertThat(registeredDeps)
        .containsExactly(
            ImmutableList.of(pKey1),
            ImmutableList.of(c1Key1),
            ImmutableList.of(gcKey1),
            ImmutableList.of(gcKey2, gcKey3),
            ImmutableList.of(c2Key1, c2Key2))
        .inOrder();
  }

  private static SkyKey createKey(String name) {
    return new SkyKey() {
      @Override
      public SkyFunctionName functionName() {
        return SkyFunctionName.createHermetic(name);
      }

      // Override toString to assist debugging.
      @Override
      public String toString() {
        return name;
      }
    };
  }
}
