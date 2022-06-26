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

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import net.starlark.java.eval.Tuple;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TupleCodec}. */
@RunWith(JUnit4.class)
public class TupleCodecTest {
  @Test
  public void testCodec() throws Exception {
    Tuple aliasedInnerTuple = Tuple.of(1, 2);
    new SerializationTester(
            Tuple.of(),
            Tuple.of(1, 2, Tuple.of(3, 4), 5),
            Tuple.of(aliasedInnerTuple, aliasedInnerTuple))
        .makeMemoizing()
        // Note that verification uses an equals() test, which ensures not just correct order, but
        // also that we got back the right kind of value (list versus tuple).
        .runTests();
  }
}
