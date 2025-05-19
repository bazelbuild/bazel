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
package com.google.devtools.common.options;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.util.regex.RegexUtil;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * Option class wrapping a {@link Pattern class}. We wrap the {@link Pattern} class instance since
 * it uses reference equality, which breaks the assumption of {@link Converter} that {@code
 * converter.convert(sameString).equals(converter.convert(sameString)}.
 *
 * <p>Please note that the equality implementation is based solely on the input regex, therefore
 * patterns expressing the same intent with different regular expressions (e.g. {@code "a"} and
 * {@code "[a]"} will not be treated as equal.
 */
@AutoValue
public abstract class RegexPatternOption {
  static RegexPatternOption create(Pattern regexPattern) {
    return new AutoValue_RegexPatternOption(
        Preconditions.checkNotNull(regexPattern),
        RegexUtil.asOptimizedMatchingPredicate(regexPattern));
  }

  /**
   * The original regex pattern.
   *
   * <p>Note: Strings passed to the {@link Pattern} and {@link java.util.regex.Matcher} API have to
   * be converted to "Unicode" form first (see {@link
   * com.google.devtools.build.lib.util.StringEncoding#internalToUnicode}.
   */
  public abstract Pattern regexPattern();

  /**
   * A potentially optimized {@link Predicate} that matches the entire input string against the
   * regex pattern.
   */
  public abstract Predicate<String> matcher();

  @Override
  public final boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof RegexPatternOption)) {
      return false;
    }

    RegexPatternOption otherOption = (RegexPatternOption) other;
    return otherOption.regexPattern().pattern().equals(regexPattern().pattern());
  }

  @Override
  public final int hashCode() {
    return regexPattern().pattern().hashCode();
  }
}
