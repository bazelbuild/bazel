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
package com.google.devtools.build.lib.collect.compacthashset;

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class CompactHashSetCodecTest {

  @Test
  public void codec_roundTrips() throws Exception {
    var setWithManyElements = CompactHashSet.<Integer>create();
    for (int i = 0; i < 1000; i++) {
      setWithManyElements.add(i);
    }
    new SerializationTester(
            CompactHashSet.create(),
            CompactHashSet.create("hello"),
            CompactHashSet.create("alpha", "beta", "gamma"),
            CompactHashSet.create("one", null, "three"),
            CompactHashSet.create("a", "b", "a"),
            setWithManyElements)
        .runTests();
  }
}
