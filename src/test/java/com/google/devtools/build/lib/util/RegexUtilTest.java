package com.google.devtools.build.lib.util;

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
