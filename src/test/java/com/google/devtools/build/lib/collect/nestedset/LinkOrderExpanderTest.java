// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 * Tests for {@link LinkOrderExpander}.
 */
@RunWith(JUnit4.class)
public class LinkOrderExpanderTest extends ExpanderTestBase {

  @Override
  protected Order expanderOrder() {
    return Order.LINK_ORDER;
  }

  @Override
  protected List<String> nestedResult() {
    return ImmutableList.of("b", "d", "c", "a", "e");
  }

  @Override
  protected List<String> nestedDuplicatesResult() {
    return ImmutableList.of("b", "d", "c", "a", "e");
  }

  @Override
  protected List<String> chainResult() {
    return ImmutableList.of("a", "b", "c");
  }

  @Override
  protected List<String> diamondResult() {
    return ImmutableList.of("a", "b", "c", "d");
  }

  @Override
  protected List<String> orderConflictResult() {
    // Rightmost branch determines the order.
    return ImmutableList.of("b", "a");
  }

  @Override
  protected List<String> extendedDiamondResult() {
    return ImmutableList.of("a", "b", "c", "e", "d");
  }

  @Override
  protected List<String> extendedDiamondRightArmResult() {
    return ImmutableList.of("a", "b", "c", "c2", "e", "d");
  }
}
