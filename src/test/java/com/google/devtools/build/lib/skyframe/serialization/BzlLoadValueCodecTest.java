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

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.BzlVisibility;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import net.starlark.java.eval.Module;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BzlLoadValue} serialization. */
@RunWith(JUnit4.class)
public class BzlLoadValueCodecTest {
  private static final ImmutableTable<RepositoryName, String, RepositoryName> SOME_TABLE =
      ImmutableTable.of(
          RepositoryName.createUnvalidated("foo"), "bar", RepositoryName.createUnvalidated("quux"));

  @Test
  public void objectCodecTests() throws Exception {
    Module module = Module.create();
    module.setGlobal("a", 1);
    module.setGlobal("b", 2);
    module.setGlobal("c", 3);
    byte[] digest = "dummy".getBytes(ISO_8859_1);

    new SerializationTester(new BzlLoadValue(module, digest, BzlVisibility.PUBLIC, SOME_TABLE))
        .setVerificationFunction(
            (SerializationTester.VerificationFunction<BzlLoadValue>)
                (x, y) -> {
                  if (!java.util.Arrays.equals(x.getTransitiveDigest(), y.getTransitiveDigest())) {
                    throw new AssertionError("unequal digests after serialization");
                  }
                  assertThat(x.getRecordedRepoMappings()).isEqualTo(y.getRecordedRepoMappings());
                })
        .runTestsWithoutStableSerializationCheck();
  }

  @Test
  public void canSendBuiltins() throws Exception {
    Object builtin = new Object();
    ObjectCodecRegistry registry =
        AutoRegistry.get().getBuilder().addReferenceConstant(builtin).build();
    BzlLoadValue value = makeBLV("var", builtin);
    BzlLoadValue deserialized = RoundTripping.roundTrip(value, registry);
    Object deserializedDummy = deserialized.getModule().getGlobal("var");
    assertThat(deserializedDummy).isSameInstanceAs(builtin);
  }

  /** Makes a simple {@link BzlLoadValue} with just one entry and no dependencies. */
  private static BzlLoadValue makeBLV(String name, Object value) {
    Module module = Module.create();
    module.setGlobal(name, value);

    byte[] digest = "dummy".getBytes(ISO_8859_1);
    return new BzlLoadValue(module, digest, BzlVisibility.PUBLIC, SOME_TABLE);
  }
}
