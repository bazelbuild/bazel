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

package com.google.devtools.build.lib.rules.java;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code JavaRuntimeInfo} */
@RunWith(JUnit4.class)
public class JavaRuntimeInfoTest {
  @Test
  public void equalityIsObjectIdentity() {
    JavaRuntimeInfo a =
        JavaRuntimeInfo.create(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            PathFragment.create(""),
            PathFragment.create(""),
            PathFragment.create(""),
            PathFragment.create(""));
    JavaRuntimeInfo b =
        JavaRuntimeInfo.create(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            PathFragment.create(""),
            PathFragment.create(""),
            PathFragment.create(""),
            PathFragment.create(""));

    new EqualsTester().addEqualityGroup(a).addEqualityGroup(b).testEquals();
  }
}
