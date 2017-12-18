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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.strings.StringCodecs;
import com.google.devtools.build.lib.skyframe.serialization.testutils.ObjectCodecTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NestedSetCodec}. */
@RunWith(JUnit4.class)
public class NestedSetCodecTest {

  private static final NestedSet<String> SHARED_NESTED_SET =
      NestedSetBuilder.<String>stableOrder().add("e").build();

  @Test
  public void testCodec() throws Exception {
    ImmutableList<NestedSet<String>> subjects =
        ImmutableList.of(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
            NestedSetBuilder.create(Order.STABLE_ORDER, "a"),
            NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b", "c"),
            NestedSetBuilder.<String>stableOrder()
                .add("a")
                .add("b")
                .addTransitive(
                    NestedSetBuilder.<String>stableOrder()
                        .add("c")
                        .addTransitive(SHARED_NESTED_SET)
                        .build())
                .addTransitive(
                    NestedSetBuilder.<String>stableOrder()
                        .add("d")
                        .addTransitive(SHARED_NESTED_SET)
                        .build())
                .addTransitive(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
                .build());

    ObjectCodecTester.newBuilder(new NestedSetCodec<>(StringCodecs.simple()))
        .addSubjects(subjects)
        .verificationFunction(NestedSetCodecTest::verifyDeserialization)
        .buildAndRunTests();
  }

  private static void verifyDeserialization(
      NestedSet<String> subject, NestedSet<String> deserialized) {
    assertThat(subject.getOrder()).isEqualTo(deserialized.getOrder());
    assertThat(subject.toSet()).isEqualTo(deserialized.toSet());
    verifyStructure(subject.rawChildren(), deserialized.rawChildren());
  }

  private static void verifyStructure(Object lhs, Object rhs) {
    if (lhs == NestedSet.EMPTY_CHILDREN) {
      assertThat(rhs).isSameAs(NestedSet.EMPTY_CHILDREN);
    } else if (lhs instanceof Object[]) {
      assertThat(rhs).isInstanceOf(Object[].class);
      Object[] lhsArray = (Object[]) lhs;
      Object[] rhsArray = (Object[]) rhs;
      int n = lhsArray.length;
      assertThat(rhsArray).hasLength(n);
      for (int i = 0; i < n; ++i) {
        verifyStructure(lhsArray[i], rhsArray[i]);
      }
    } else {
      assertThat(lhs).isEqualTo(rhs);
    }
  }
}
