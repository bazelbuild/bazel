// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.License.LicenseType;
import java.util.Arrays;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class LicenseTest {

  @Test
  public void testLeastRestrictive() {
    assertThat(License.leastRestrictive(Arrays.asList(LicenseType.RESTRICTED)))
        .isEqualTo(LicenseType.RESTRICTED);
    assertThat(
            License.leastRestrictive(
                Arrays.asList(LicenseType.RESTRICTED, LicenseType.BY_EXCEPTION_ONLY)))
        .isEqualTo(LicenseType.RESTRICTED);
    assertThat(License.leastRestrictive(Arrays.<LicenseType>asList()))
        .isEqualTo(LicenseType.BY_EXCEPTION_ONLY);
  }

  /** Evaluates a string as a Starlark expression returning a sequence of strings. */
  private static Sequence<String> evalAsSequence(String string) throws Exception {
    ParserInput input = ParserInput.fromLines(string);
    Mutability mutability = Mutability.create("test");
    Object parsedValue =
        Starlark.execFile(
            input,
            FileOptions.DEFAULT,
            Module.create(),
            new StarlarkThread(mutability, StarlarkSemantics.DEFAULT));
    mutability.freeze();
    return Sequence.cast(parsedValue, String.class, "evalAsSequence() input");
  }

  @Test
  public void repr() throws Exception {
    assertThat(Starlark.repr(License.NO_LICENSE)).isEqualTo("[\"none\"]");
    assertThat(License.parseLicense(evalAsSequence(Starlark.repr(License.NO_LICENSE))))
        .isEqualTo(License.NO_LICENSE);

    License withoutExceptions = License.parseLicense(ImmutableList.of("notice", "restricted"));
    // License types sorted by LicenseType enum order.
    assertThat(Starlark.repr(withoutExceptions)).isEqualTo("[\"restricted\", \"notice\"]");
    assertThat(License.parseLicense(evalAsSequence(Starlark.repr(withoutExceptions))))
        .isEqualTo(withoutExceptions);

    License withExceptions =
        License.parseLicense(
            ImmutableList.of("notice", "restricted", "exception=//foo:bar", "exception=//baz:qux"));
    // Exceptions sorted alphabetically.
    assertThat(Starlark.repr(withExceptions))
        .isEqualTo(
            "[\"restricted\", \"notice\", \"exception=//baz:qux\", \"exception=//foo:bar\"]");
    assertThat(License.parseLicense(evalAsSequence(Starlark.repr(withExceptions))))
        .isEqualTo(withExceptions);
  }
}
