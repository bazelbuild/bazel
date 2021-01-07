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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link BzlCompileValue.Key} serialization. */
@RunWith(JUnit4.class)
public final class BzlCompileKeyCodecTest {
  @Test
  public void testCodec() throws Exception {
    Root root = FsUtils.TEST_ROOTED_PATH.getRoot();
    SerializationTester serializationTester =
        new SerializationTester(
            BzlCompileValue.EMPTY_PRELUDE_KEY,
            BzlCompileValue.keyForBuildPrelude(root, Label.parseAbsoluteUnchecked("//a:a")),
            BzlCompileValue.key(root, Label.parseAbsoluteUnchecked("//a:a")));
    FsUtils.addDependencies(serializationTester);
    // Indirectly test that deserialization does interning by verifying that the deserialized
    // instance is the same as the serialized one.
    serializationTester.setVerificationFunction((in, out) -> assertThat(out).isSameInstanceAs(in));
    serializationTester.runTests();
  }
}
