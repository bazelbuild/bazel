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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TransformingRandomAccessListCodecTest {

  @Test
  public void codec_roundtrips() throws Exception {
    new SerializationTester(
            Lists.transform(ImmutableList.of(), input -> input),
            Lists.transform(ImmutableList.of("abc", "def"), input -> input + "_suffix"))
        // Serialization transforms the TransformingRandomAccessList into an equivalent
        // ImmutableList. So serializing it again results in a different tag value.
        .runTestsWithoutStableSerializationCheck();
  }
}
