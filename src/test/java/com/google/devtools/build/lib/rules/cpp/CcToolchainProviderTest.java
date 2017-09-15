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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@code CcToolchainProvider}
 */
@RunWith(JUnit4.class)
public class CcToolchainProviderTest {
  @Test
  public void equalityIsObjectIdentity() throws Exception {
    CcToolchainProvider a = new CcToolchainProvider(
        null,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        PathFragment.EMPTY_FRAGMENT,
        CppCompilationContext.EMPTY,
        false,
        false,
        ImmutableMap.<String, String>of(),
        ImmutableList.<Artifact>of(),
        NestedSetBuilder.<Pair<String, String>>emptySet(Order.COMPILE_ORDER),
        null,
        ImmutableMap.<String, String>of(),
        ImmutableList.<PathFragment>of(),
        null);

    CcToolchainProvider b = new CcToolchainProvider(
        null,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        PathFragment.EMPTY_FRAGMENT,
        CppCompilationContext.EMPTY,
        false,
        false,
        ImmutableMap.<String, String>of(),
        ImmutableList.<Artifact>of(),
        NestedSetBuilder.<Pair<String, String>>emptySet(Order.COMPILE_ORDER),
        null,
        ImmutableMap.<String, String>of(),
        ImmutableList.<PathFragment>of(),
        null);

    new EqualsTester()
        .addEqualityGroup(a)
        .addEqualityGroup(b)
        .testEquals();
  }
}
