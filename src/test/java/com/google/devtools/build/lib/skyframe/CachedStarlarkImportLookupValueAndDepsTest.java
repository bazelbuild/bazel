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
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CachedStarlarkImportLookupValueAndDeps}. */
@RunWith(JUnit4.class)
public class CachedStarlarkImportLookupValueAndDepsTest {
  @Test
  public void testDepsAreNotVisitedMultipleTimesForDiamondDependencies() throws Exception {
    // Graph structure of StarlarkImportLookupValues:
    //
    //     p
    //   /  \
    //  c1  c2
    //   \  /
    //    gc

    StarlarkImportLookupValue dummyValue = mock(StarlarkImportLookupValue.class);
    CachedStarlarkImportLookupValueAndDepsBuilderFactory
        cachedStarlarkImportLookupValueAndDepsBuilderFactory =
            new CachedStarlarkImportLookupValueAndDepsBuilderFactory();

    StarlarkImportLookupValue.Key gcKey = createStarlarkKey("//gc");
    SkyKey gcKey1 = createKey("gc key1");
    SkyKey gcKey2 = createKey("gc key2");
    SkyKey gcKey3 = createKey("gc key3");
    CachedStarlarkImportLookupValueAndDeps gc =
        cachedStarlarkImportLookupValueAndDepsBuilderFactory
            .newCachedStarlarkImportLookupValueAndDepsBuilder()
            .addDep(gcKey1)
            .addDeps(ImmutableList.of(gcKey2, gcKey3))
            .setKey(gcKey)
            .setValue(dummyValue)
            .build();

    StarlarkImportLookupValue.Key c1Key = createStarlarkKey("//c1");
    SkyKey c1Key1 = createKey("c1 key1");
    CachedStarlarkImportLookupValueAndDeps c1 =
        cachedStarlarkImportLookupValueAndDepsBuilderFactory
            .newCachedStarlarkImportLookupValueAndDepsBuilder()
            .addDep(c1Key1)
            .addTransitiveDeps(gc)
            .setValue(dummyValue)
            .setKey(c1Key)
            .build();

    StarlarkImportLookupValue.Key c2Key = createStarlarkKey("//c2");
    SkyKey c2Key1 = createKey("c2 key1");
    SkyKey c2Key2 = createKey("c2 key2");
    CachedStarlarkImportLookupValueAndDeps c2 =
        cachedStarlarkImportLookupValueAndDepsBuilderFactory
            .newCachedStarlarkImportLookupValueAndDepsBuilder()
            .addDeps(ImmutableList.of(c2Key1, c2Key2))
            .addTransitiveDeps(gc)
            .setValue(dummyValue)
            .setKey(c2Key)
            .build();

    StarlarkImportLookupValue.Key pKey = createStarlarkKey("//p");
    SkyKey pKey1 = createKey("p key1");
    CachedStarlarkImportLookupValueAndDeps p =
        cachedStarlarkImportLookupValueAndDepsBuilderFactory
            .newCachedStarlarkImportLookupValueAndDepsBuilder()
            .addDep(pKey1)
            .addTransitiveDeps(c1)
            .addTransitiveDeps(c2)
            .setValue(dummyValue)
            .setKey(pKey)
            .build();

    List<Iterable<SkyKey>> registeredDeps = new ArrayList<>();
    Map<StarlarkImportLookupValue.Key, CachedStarlarkImportLookupValueAndDeps>
        visitedDepsInToplevelLoad = new HashMap<>();
    p.traverse(registeredDeps::add, visitedDepsInToplevelLoad);

    assertThat(registeredDeps)
        .containsExactly(
            ImmutableList.of(pKey1),
            ImmutableList.of(c1Key1),
            ImmutableList.of(gcKey1),
            ImmutableList.of(gcKey2, gcKey3),
            ImmutableList.of(c2Key1, c2Key2))
        .inOrder();

    // Note that (pKey, p) is expected to be added separately.
    assertThat(visitedDepsInToplevelLoad).containsExactly(c1Key, c1, c2Key, c2, gcKey, gc);
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

  private static StarlarkImportLookupValue.Key createStarlarkKey(String name) {
    return StarlarkImportLookupValue.packageBzlKey(Label.parseAbsoluteUnchecked(name));
  }
}
