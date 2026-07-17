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
package com.google.devtools.build.lib.util.regex;

import static com.google.common.truth.Truth.assertThat;

import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link RegexUtil}. */
@RunWith(TestParameterInjector.class)
public class RegexUtilTest {

  @Test
  public void optimizedMatchingPredicate(
      @TestParameter({
            "",
            ".",
            "a",
            "foo",
            "foofoo",
            "coverage.dat",
            "/coverage.dat",
            "/coverage.data",
            "/coverage1dat",
            "/coverage1data",
            "foo/coverage.dat",
            "foo/coverage.data",
            "foo/coverage1dat",
            "foo/coverage1data",
            "foo/test/a/coverage.dat",
            "foo/test/.*/coverage.dat",
            "]]\n",
            "()",
            "+",
            "|",
          })
          String haystack,
      @TestParameter({
            ".*",
            ".*?foo",
            ".*+foo",
            "^foo$",
            "coverage\\.dat",
            ".*/coverage.dat",
            ".*/coverage\\.dat",
            ".*/test/.*/coverage\\.dat",
            "$|",
            "^",
            ".]",
            ".*]",
            ".*^?^\\Q",
            "foo|/coverage.dat",
            ".*^|.*a",
            "\\Q.",
            ".*.",
            ".*\\\\",
            ".*()",
            ".*|",
            ".*^",
            ".*+",
          })
          String needle) {
    Pattern originalPattern = Pattern.compile(needle, Pattern.DOTALL);
    Predicate<String> optimizedMatcher = RegexUtil.asOptimizedMatchingPredicate(originalPattern);
    assertThat(optimizedMatcher.test(haystack))
        .isEqualTo(originalPattern.matcher(haystack).matches());
  }
}
