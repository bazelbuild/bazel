// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternsValue.TargetPatternSequence;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for serialization of {@link TargetPatternSequence}. */
@RunWith(JUnit4.class)
public final class TargetPatternSequenceCodecTest {
  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            TargetPatternSequence.create(ImmutableList.of(), PathFragment.EMPTY_FRAGMENT),
            TargetPatternSequence.create(
                ImmutableList.of("foo", "bar"), PathFragment.create("baz")),
            TargetPatternSequence.create(
                ImmutableList.of("uno", "dos"), PathFragment.create("tres")),
            TargetPatternSequence.create(
                ImmutableList.of("dos", "uno"), PathFragment.create("tres")))
        .runTests();
  }

  @Test
  public void testPatternsOrderSignificant() throws Exception {
    ObjectCodecs codecs = new ObjectCodecs();
    byte[] serialized1 =
        codecs
            .serialize(
                TargetPatternSequence.create(
                    ImmutableList.of("uno", "dos"), PathFragment.create("tres")))
            .toByteArray();
    assertThat(serialized1).asList().isNotEmpty();

    byte[] serialized2 =
        codecs
            .serialize(
                TargetPatternSequence.create(
                    ImmutableList.of("dos", "uno"), PathFragment.create("tres")))
            .toByteArray();
    assertThat(serialized2).asList().isNotEmpty();
    assertThat(serialized1).isNotEqualTo(serialized2);
  }
}
