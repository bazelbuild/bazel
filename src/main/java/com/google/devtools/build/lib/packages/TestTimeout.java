// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableRangeMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Range;
import com.google.common.collect.RangeMap;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Symbolic labels of test timeout. Borrows heavily from {@link TestSize}.
 */
public enum TestTimeout {

  // These symbolic labels are used in the build files.
  SHORT(60),
  MODERATE(300),
  LONG(900),
  ETERNAL(3600);

  /**
   * Default --test_timeout flag, used when collecting code coverage.
   */
  public static final String COVERAGE_CMD_TIMEOUT = "--test_timeout=300,600,1200,3600";

  /** Map from test time to suggested TestTimeout. */
  private static final RangeMap<Integer, TestTimeout> SUGGESTED_TIMEOUT;

  /**
   * Map from TestTimeout to fuzzy range.
   *
   * <p>The fuzzy range is used to check whether the actual timeout is close to the upper bound of
   * the current timeout or much smaller than the next shorter timeout. This is used to give
   * suggestions to developers to update their timeouts.
   */
  private static final Map<TestTimeout, Range<Integer>> TIMEOUT_FUZZY_RANGE;

  static {
    // For the largest timeout, cap suggested and fuzzy ranges at one year.
    final int maxTimeout = 365 * 24 * 60 * 60 /* One year */;

    ImmutableRangeMap.Builder<Integer, TestTimeout> suggestedTimeoutBuilder =
        ImmutableRangeMap.builder();
    ImmutableMap.Builder<TestTimeout, Range<Integer>> timeoutFuzzyRangeBuilder =
        ImmutableMap.builder();

    int previousMaxSuggested = 0;
    int previousTimeout = 0;

    Iterator<TestTimeout> timeoutIterator = Arrays.asList(values()).iterator();
    while (timeoutIterator.hasNext()) {
      TestTimeout timeout = timeoutIterator.next();

      // Set up time ranges for suggested timeouts and fuzzy timeouts. Fuzzy timeout ranges should
      // be looser than suggested timeout ranges in order to make sure that after a test size is
      // adjusted, it's difficult for normal time variance to push it outside the fuzzy timeout
      // range.

      // This should be exactly the previous max because there should be exactly one suggested
      // timeout for any given time.
      final int minSuggested = previousMaxSuggested;
      // Only suggest timeouts that are less than 75% of the actual timeout (unless there are no
      // higher timeouts). This should be low enough to prevent suggested times from causing test
      // timeout flakiness.
      final int maxSuggested =
          timeoutIterator.hasNext() ? (int) (timeout.timeout * 0.75) : maxTimeout;

      // Set fuzzy minimum timeout to half the previous timeout. If the test is that fast, it should
      // be safe to use the shorter timeout.
      final int minFuzzy = previousTimeout / 2;
      // Set fuzzy maximum timeout to 90% of the timeout. A test this close to the limit can easily
      // become timeout flaky.
      final int maxFuzzy = timeoutIterator.hasNext() ? (int) (timeout.timeout * 0.9) : maxTimeout;

      timeoutFuzzyRangeBuilder.put(timeout, Range.closedOpen(minFuzzy, maxFuzzy));

      suggestedTimeoutBuilder.put(Range.closedOpen(minSuggested, maxSuggested), timeout);

      previousMaxSuggested = maxSuggested;
      previousTimeout = timeout.timeout;
    }
    SUGGESTED_TIMEOUT = suggestedTimeoutBuilder.build();
    TIMEOUT_FUZZY_RANGE = timeoutFuzzyRangeBuilder.build();
  }

  private final int timeout;

  TestTimeout(int timeout) {
    this.timeout = timeout;
  }

  /**
   * Returns the enum associated with a test's timeout or null if the tag is
   * not lower case or an unknown size.
   */
  public static TestTimeout getTestTimeout(String attr) {
    if (!attr.equals(attr.toLowerCase())) {
      return null;
    }
    try {
      return TestTimeout.valueOf(attr.toUpperCase(Locale.ENGLISH));
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  @Override
  public String toString() {
    return super.toString().toLowerCase();
  }

  /**
   * We print to upper case to make the test timeout warnings more readable.
   */
  public String prettyPrint() {
    return super.toString().toUpperCase();
  }

  @Deprecated // use getTimeout instead
  public int getTimeoutSeconds() {
    return timeout;
  }

  public Duration getTimeout() {
    return Duration.ofSeconds(timeout);
  }

  /**
   * Returns true iff the given time is not close to the upper bound timeout and is so short that it
   * should be assigned a different timeout.
   *
   * <p>This is used to give suggestions to developers to update their timeouts. If this returns
   * true, a more reasonable timeout can be selected with {@link #getSuggestedTestTimeout(int)}
   */
  public boolean isInRangeFuzzy(int timeInSeconds) {
    return TIMEOUT_FUZZY_RANGE.get(this).contains(timeInSeconds);
  }

  /**
   * Returns suggested test size for the given time in seconds.
   *
   * <p>Will suggest times that are unlikely to result in timeout flakiness even if the test has a
   * significant amount of time variance.
   */
  public static TestTimeout getSuggestedTestTimeout(int timeInSeconds) {
    return SUGGESTED_TIMEOUT.get(timeInSeconds);
  }

  /**
   * Returns test timeout of the given test target using explicitly specified timeout
   * or default through to the size label's associated default.
   */
  public static TestTimeout getTestTimeout(Rule testTarget) {
    String attr = NonconfigurableAttributeMapper.of(testTarget).get("timeout", Type.STRING);
    if (!attr.equals(attr.toLowerCase())) {
      return null;  // attribute values must be lowercase
    }
    try {
      return TestTimeout.valueOf(attr.toUpperCase(Locale.ENGLISH));
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  /**
   * Converter for the --test_timeout option.
   */
  public static class TestTimeoutConverter implements Converter<Map<TestTimeout, Duration>> {
    public TestTimeoutConverter() {}

    @Override
    public Map<TestTimeout, Duration> convert(String input) throws OptionsParsingException {
      List<Duration> values = new ArrayList<>();
      for (String token : Splitter.on(',').limit(6).split(input)) {
        // Handle the case of "2," which is accepted as legal... Because Splitter.split is lazy,
        // there's no way of knowing if an empty string is a trailing or an intermediate one,
        // so we can't fully emulate String.split(String, 0).
        if (!token.isEmpty() || values.size() > 1) {
          try {
            values.add(Duration.ofSeconds(Integer.parseInt(token)));
          } catch (NumberFormatException e) {
            throw new OptionsParsingException("'" + input + "' is not an int");
          }
        }
      }
      EnumMap<TestTimeout, Duration> timeouts = Maps.newEnumMap(TestTimeout.class);
      if (values.size() == 1) {
        timeouts.put(SHORT, values.get(0));
        timeouts.put(MODERATE, values.get(0));
        timeouts.put(LONG, values.get(0));
        timeouts.put(ETERNAL, values.get(0));
      } else if (values.size() == 4) {
        timeouts.put(SHORT, values.get(0));
        timeouts.put(MODERATE, values.get(1));
        timeouts.put(LONG, values.get(2));
        timeouts.put(ETERNAL, values.get(3));
      } else {
        throw new OptionsParsingException("Invalid number of comma-separated entries");
      }
      for (TestTimeout label : values()) {
        if (!timeouts.containsKey(label) || timeouts.get(label).compareTo(Duration.ZERO) <= 0) {
          timeouts.put(label, label.getTimeout());
        }
      }
      return timeouts;
    }

    @Override
    public String getTypeDescription() {
      return "a single integer or comma-separated list of 4 integers";
    }
  }

  /**
   * Converter for the --test_timeout_filters option.
   */
  public static class TestTimeoutFilterConverter extends EnumFilterConverter<TestTimeout> {
    public TestTimeoutFilterConverter() {
      super(TestTimeout.class, "test timeout");
    }

    /**
     * {@inheritDoc}
     *
     * <p>This override is necessary to prevent OptionsData
     * from throwing a "must be assignable from the converter return type" exception.
     * OptionsData doesn't recognize the generic type and actual type are the same.
     */
    @Override
    public final Set<TestTimeout> convert(String input) throws OptionsParsingException {
      return super.convert(input);
    }
  }
}
