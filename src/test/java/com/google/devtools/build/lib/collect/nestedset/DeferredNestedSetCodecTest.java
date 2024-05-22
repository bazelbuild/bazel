// Copyright 2024 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructure;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.AutoRegistry;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationResult;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class DeferredNestedSetCodecTest {
  @Test
  public void empty() throws Exception {
    new SerializationTester(
            Order.STABLE_ORDER.emptySet(),
            Order.COMPILE_ORDER.emptySet(),
            Order.LINK_ORDER.emptySet(),
            Order.NAIVE_LINK_ORDER.emptySet())
        .addCodec(new DeferredNestedSetCodec())
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .runTests();
  }

  @Test
  public void singleton() throws Exception {
    new SerializationTester(
            NestedSetBuilder.stableOrder().add("A").build(),
            NestedSetBuilder.compileOrder().add("B").build(),
            NestedSetBuilder.linkOrder().add("C").build(),
            NestedSetBuilder.naiveLinkOrder().add("D").build())
        .addCodec(new DeferredNestedSetCodec())
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(DeferredNestedSetCodecTest::verifyUsingShallowEquals)
        .runTests();
  }

  @Test
  public void array() throws Exception {
    new SerializationTester(
            NestedSetBuilder.stableOrder().addAll(ImmutableList.of(1, 2, 3)).build(),
            NestedSetBuilder.compileOrder().addAll(ImmutableList.of("A", "B", "C")).build(),
            NestedSetBuilder.linkOrder().addAll(ImmutableList.of(5.56, 3.14, 10, 20)).build(),
            NestedSetBuilder.naiveLinkOrder()
                .addAll(ImmutableList.of("one", "two", "three", "four", "five"))
                .build())
        .addCodec(new DeferredNestedSetCodec())
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .setVerificationFunction(DeferredNestedSetCodecTest::verifyUsingShallowEquals)
        .runTests();
  }

  @Test
  public void diamond() throws Exception {
    var root = NestedSetBuilder.stableOrder().addAll(ImmutableList.of(1, 2)).build();
    var left = NestedSetBuilder.stableOrder().add("left").addTransitive(root).build();
    var right = NestedSetBuilder.stableOrder().add("right").addTransitive(root).build();
    var top =
        NestedSetBuilder.stableOrder()
            .addAll(ImmutableList.of("this", "is", "the", "top"))
            .addTransitive(left)
            .addTransitive(right)
            .build();

    var fingerprintValueService = FingerprintValueService.createForTesting();
    var codecs =
        new ObjectCodecs(AutoRegistry.get().getBuilder().add(new DeferredNestedSetCodec()).build());

    SerializationResult<ByteString> serialized =
        codecs.serializeMemoizedAndBlocking(fingerprintValueService, top);
    ListenableFuture<Void> futureToBlockWritesOn = serialized.getFutureToBlockWritesOn();
    if (futureToBlockWritesOn != null) {
      var unused = futureToBlockWritesOn.get();
    }
    ByteString bytes = serialized.getObject();

    NestedSet<?> deserialized =
        (NestedSet<?>) codecs.deserializeMemoizedAndBlocking(fingerprintValueService, bytes);
    // Since dumpStructure doesn't perform equivalence reduction, equivalence here means the diamond
    // reference structure was preserved by deserialization.
    assertThat(dumpStructure(top)).isEqualTo(dumpStructure(deserialized));
  }

  @SuppressWarnings({"rawtypes", "unchecked"})
  private static void verifyUsingShallowEquals(NestedSet original, NestedSet deserialized) {
    assertThat(original.shallowEquals(deserialized)).isTrue();
  }
}
