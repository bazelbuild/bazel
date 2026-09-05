// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import com.google.common.base.Splitter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import java.util.Objects;

/** Parsed value of {@code --flaky_test_attempts}. */
public final class FlakyTestAttempts {
  public static final FlakyTestAttempts DEFAULT = new FlakyTestAttempts(1, 3);

  private final int stableAttempts;
  private final int flakyAttempts;

  public FlakyTestAttempts(int stableAttempts, int flakyAttempts) {
    this.stableAttempts = stableAttempts;
    this.flakyAttempts = flakyAttempts;
  }

  public int getStableAttempts() {
    return stableAttempts;
  }

  public int getFlakyAttempts() {
    return flakyAttempts;
  }

  public int getAttempts(boolean isFlaky) {
    return isFlaky ? flakyAttempts : stableAttempts;
  }

  /** Returns the canonical string stored in {@link PerLabelOptions}. */
  public String toCanonicalString() {
    if (stableAttempts == flakyAttempts) {
      return Integer.toString(stableAttempts);
    }
    return stableAttempts + "," + flakyAttempts;
  }

  public static FlakyTestAttempts parse(String input) throws OptionsParsingException {
    if (Objects.equals(input, "default")) {
      return DEFAULT;
    }
    List<String> parts = Splitter.on(',').splitToList(input);
    if (parts.size() == 1) {
      int attempts = parseAttemptCount(parts.get(0), input);
      return new FlakyTestAttempts(attempts, attempts);
    }
    if (parts.size() == 2) {
      return new FlakyTestAttempts(
          parseAttemptCount(parts.get(0), input), parseAttemptCount(parts.get(1), input));
    }
    throw new OptionsParsingException(
        "'"
            + input
            + "' must be 'default', a single integer, or two comma-separated integers"
            + " (<stable>,<flaky>)");
  }

  private static int parseAttemptCount(String token, String originalInput)
      throws OptionsParsingException {
    if (token.isEmpty()) {
      throw new OptionsParsingException(
          "'"
              + originalInput
              + "' must be 'default', a single integer, or two comma-separated integers"
              + " (<stable>,<flaky>)");
    }
    try {
      return validateAttemptCount(Integer.parseInt(token), originalInput);
    } catch (NumberFormatException e) {
      throw new OptionsParsingException("'" + originalInput + "' is not an integer", e);
    }
  }

  static int validateAttemptCount(int value, String originalInput) throws OptionsParsingException {
    if (value < ExecutionOptions.TestAttemptsConverter.MIN_VALUE) {
      throw new OptionsParsingException(
          "'"
              + originalInput
              + "' should be >= "
              + ExecutionOptions.TestAttemptsConverter.MIN_VALUE);
    }
    if (value > ExecutionOptions.TestAttemptsConverter.MAX_VALUE) {
      throw new OptionsParsingException(
          "'"
              + originalInput
              + "' should be <= "
              + ExecutionOptions.TestAttemptsConverter.MAX_VALUE);
    }
    return value;
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof FlakyTestAttempts other)) {
      return false;
    }
    return stableAttempts == other.stableAttempts && flakyAttempts == other.flakyAttempts;
  }

  @Override
  public int hashCode() {
    return Objects.hash(stableAttempts, flakyAttempts);
  }
}
