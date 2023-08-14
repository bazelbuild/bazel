// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ModuleCodec}. */
@RunWith(JUnit4.class)
public class ModuleCodecTest {
  @Test
  public void testCodec() throws Exception {
    Module subject1 = Module.create();

    Module subject2 =
        Module.withPredeclaredAndData(
            StarlarkSemantics.DEFAULT, ImmutableMap.of(), Label.parseCanonical("//foo:bar"));
    subject2.setGlobal("x", 1);
    subject2.setGlobal("y", 2);

    new SerializationTester(subject1, subject2)
        .makeMemoizing()
        .setVerificationFunction(ModuleCodecTest::verifyDeserialization)
        .runTestsWithoutStableSerializationCheck();
  }

  private static void verifyDeserialization(Module subject, Module deserialized) {
    // Module doesn't implement proper equality.
    TestUtils.assertModulesEqual(subject, deserialized);
  }
}
