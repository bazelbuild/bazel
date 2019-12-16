// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest.functional;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.flags.Flag;
import com.google.common.flags.FlagSpec;
import com.google.common.flags.Flags;
import com.google.devtools.build.android.desugar.DesugarRule;
import com.google.devtools.build.android.desugar.DesugarRule.LoadClass;
import com.google.testing.junit.junit4.api.TestArgs;
import com.google.testing.testsize.MediumTest;
import com.google.testing.testsize.MediumTestAttribute;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for accessing a series of private fields, constructors and methods from another class
 * within a nest.
 */
@RunWith(JUnit4.class)
@MediumTest(MediumTestAttribute.FILE)
public final class ComplexCasesTest {

  @FlagSpec(
      help = "The root directory of source root directory for desugar testing.",
      name = "test_jar_complexcase")
  private static final Flag<String> testJar = Flag.nullString();

  private static final MethodHandles.Lookup lookup = MethodHandles.lookup();

  @Rule
  public final DesugarRule desugarRule =
      DesugarRule.builder(this, lookup)
          .addRuntimeInputs(testJar.getNonNull())
          .addCommandOptions("desugar_nest_based_private_access", "true")
          .build();

  @LoadClass("complexcase.Xylem")
  private Class<?> invoker;

  @BeforeClass
  public static void setUpFlags() throws Exception {
    Flags.parse(TestArgs.get());
  }

  @Test
  public void comprehensiveTest() throws Throwable {
    long x = 2L;
    int y = 3;
    MethodHandle testHandle =
        lookup.findStatic(
            invoker, "execute", MethodType.methodType(long.class, long.class, int.class));
    long result = (long) testHandle.invoke(x, y);

    assertThat(result).isEqualTo(14004004171L);
  }
}
