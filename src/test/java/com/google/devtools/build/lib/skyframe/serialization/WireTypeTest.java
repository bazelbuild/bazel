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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.WireType.Backreference;
import com.google.devtools.build.lib.skyframe.serialization.WireType.CodecWireType;
import com.google.devtools.build.lib.skyframe.serialization.WireType.ConstantWireType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class WireTypeTest {

  @Test
  public void specialTypedTags_matchExpectedEncoding() {
    assertThat(WireType.NULL_TYPED_TAG).isEqualTo(0);
    assertThat(WireType.LEAF_IMMEDIATE_TYPED_TAG).isEqualTo(8);
    assertThat(CodecWireType.UNSTABLE.getTypedTagNumber(0)).isEqualTo(WireType.NULL_TYPED_TAG);
    assertThat(CodecWireType.UNSTABLE.getTypedTagNumber(1))
        .isEqualTo(WireType.LEAF_IMMEDIATE_TYPED_TAG);
  }

  @Test
  public void typedTagEncodingAndDecoding_roundTripsAcrossAllWireTypes() {
    ImmutableList<WireType> allWireTypes =
        ImmutableList.of(
            CodecWireType.UNSTABLE,
            ConstantWireType.UNSTABLE,
            Backreference.INSTANCE,
            CodecWireType.STABLE_PUBLIC,
            ConstantWireType.STABLE_PUBLIC,
            CodecWireType.STABLE_PRIVATE,
            ConstantWireType.STABLE_PRIVATE);

    for (WireType wireType : allWireTypes) {
      int typedTag = wireType.getTypedTagNumber(42);
      assertThat(WireType.getTagNumber(typedTag)).isEqualTo(42);
      assertThat(WireType.getWireTypeIndex(typedTag)).isEqualTo(wireType.getValue());
      assertThat(WireType.fromValue(WireType.getWireTypeIndex(typedTag))).isEqualTo(wireType);
      assertThat(wireType.matches(typedTag)).isTrue();
      for (WireType other : allWireTypes) {
        if (other != wireType) {
          assertThat(other.matches(typedTag)).isFalse();
        }
      }
    }
  }

  @Test
  public void matches_unassignedWireTypeTag_returnsFalse() {
    int unassignedWireTypeTag = 0b111;
    assertThat(ConstantWireType.STABLE_PRIVATE.matches(unassignedWireTypeTag)).isFalse();
  }

  @Test
  public void fromValue_unassignedWireTypeValue_throwsIllegalArgumentException() {
    assertThrows(IllegalArgumentException.class, () -> WireType.fromValue(0b111));
  }
}
