// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.Maps.immutableEntry;
import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ImmutableEntryCodec}. */
@RunWith(JUnit4.class)
public class ImmutableEntryCodecTest {

  @Test
  public void testStringStringEntry_roundTripsSuccessfully() throws Exception {
    Map.Entry<String, String> original = immutableEntry("foo", "bar");
    new SerializationTester(original)
        .setVerificationFunction(
            (in, out) -> {
              assertThat(out).isEqualTo(in);
              // Verify it's the same specific class type.
              assertThat(out.getClass()).isEqualTo(in.getClass());
            })
        .runTests();
  }

  @Test
  public void roundTripsSuccessfully() throws Exception {
    new SerializationTester(
            immutableEntry(123, "baz"),
            immutableEntry(null, "value"),
            immutableEntry("key", null),
            immutableEntry(null, null))
        .runTests();
  }

  @Test
  public void testNestedEntry_roundTripsSuccessfully() throws Exception {
    Map.Entry<String, Map.Entry<Integer, String>> original =
        immutableEntry("outer", immutableEntry(1, "inner"));
    new SerializationTester(original)
        .setVerificationFunction(
            (in, out) -> {
              assertThat(out).isEqualTo(in);
              assertThat(out.getClass()).isEqualTo(in.getClass());
              assertThat(((Map.Entry) out).getValue().getClass())
                  .isEqualTo(((Map.Entry) in).getValue().getClass());
            })
        .runTests();
  }
}
