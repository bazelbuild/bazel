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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Objects;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester.VerificationFunction;
import java.io.IOException;

/** Utilities for testing NestedSet serialization. */
public class NestedSetCodecTestUtils {

  private static final NestedSet<String> SHARED_NESTED_SET =
      NestedSetBuilder.<String>stableOrder().add("e").build();

  private static class HasNestedSet {
    private final NestedSet<String> nestedSetField;

    HasNestedSet(NestedSet<String> nestedSetField) {
      this.nestedSetField = nestedSetField;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      HasNestedSet that = (HasNestedSet) o;
      return Objects.equal(nestedSetField.getChildren(), that.nestedSetField.getChildren());
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(nestedSetField);
    }
  }

  /** Perform serialization/deserialization checks for several simple NestedSet examples. */
  public static void checkCodec(
      ObjectCodecs objectCodecs, boolean allowFutureBlocking, boolean assertSymmetricEquality)
      throws Exception {
    new SerializationTester(
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
                .build(),
            NestedSetBuilder.create(
                Order.STABLE_ORDER,
                new HasNestedSet(NestedSetBuilder.create(Order.STABLE_ORDER, "a"))))
        .setObjectCodecs(objectCodecs)
        .makeMemoizingAndAllowFutureBlocking(allowFutureBlocking)
        .setVerificationFunction(verificationFunction(assertSymmetricEquality))
        .runTests();
  }

  public static ListenableFuture<Void> writeToStoreFuture(
      NestedSetStore store, NestedSet<?> nestedSet, SerializationContext serializationContext)
      throws IOException, SerializationException {
    return store
        .computeFingerprintAndStore((Object[]) nestedSet.getChildren(), serializationContext)
        .writeStatus();
  }

  private static VerificationFunction<NestedSet<String>> verificationFunction(
      boolean assertSymmetricEquality) {
    return (subject, deserialized) -> {
      if (assertSymmetricEquality) {
        assertThat(deserialized).isEqualTo(subject);
        assertThat(subject).isEqualTo(deserialized);
      }
      assertThat(subject.getOrder()).isEqualTo(deserialized.getOrder());
      assertThat(subject.toSet()).isEqualTo(deserialized.toSet());
      verifyStructure(subject.getChildren(), deserialized.getChildren());
    };
  }

  private static void verifyStructure(Object lhs, Object rhs) {
    if (lhs instanceof Object[] lhsArray) {
      assertThat(rhs).isInstanceOf(Object[].class);
      Object[] rhsArray = (Object[]) rhs;
      int n = lhsArray.length;
      assertThat(rhsArray).hasLength(n);
      for (int i = 0; i < n; ++i) {
        verifyStructure(lhsArray[i], rhsArray[i]);
      }
      if (lhsArray.length == 0) {
        // Verify empty-children is optimized - we're not creating multiple empty arrays.
        assertThat(lhsArray).isSameInstanceAs(rhsArray);
      }
    } else {
      assertThat(lhs).isEqualTo(rhs);
    }
  }
}
