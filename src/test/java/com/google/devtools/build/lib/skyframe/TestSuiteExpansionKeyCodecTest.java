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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.TestsForTargetPatternValue.TestsForTargetPatternKey;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Serialization test for test_suite. */
@RunWith(JUnit4.class)
public final class TestSuiteExpansionKeyCodecTest {

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            new TestsForTargetPatternKey(
                ImmutableSortedSet.of(
                    Label.parseAbsolute("//foo/bar:baz", ImmutableMap.of()),
                    Label.parseAbsolute("//a/b:c", ImmutableMap.of()))),
            new TestsForTargetPatternKey(ImmutableSortedSet.<Label>of()))
        .runTests();
  }
}
