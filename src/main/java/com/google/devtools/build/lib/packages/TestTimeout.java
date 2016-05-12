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
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Symbolic labels of test timeout. Borrows heavily from {@link TestSize}.
 */
public enum TestTimeout {

  // These symbolic labels are used in the build files.
  SHORT(0, 60, 60),
  MODERATE(30, 300, 300),
  LONG(300, 900, 900),
  ETERNAL(900, 365 * 24 * 60 * 60 /* One year */, 3600);

  /**
   * Default --test_timeout flag, used when collecting code coverage.
   */
  public static final String COVERAGE_CMD_TIMEOUT = "--test_timeout=300,600,1200,3600";

  private final Integer rangeMin;
  private final Integer rangeMax;
  private final Integer timeout;

  private TestTimeout(Integer rangeMin, Integer rangeMax, Integer timeout) {
    this.rangeMin = rangeMin;
    this.rangeMax = rangeMax;
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
      return TestTimeout.valueOf(attr.toUpperCase());
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

  public Integer getTimeout() {
    return timeout;
  }
  /**
   * Returns true iff the given time in seconds is exactly in the range of valid
   * execution times for this TestSize.
   */
  public boolean isInRangeExact(Integer timeInSeconds) {
    return timeInSeconds >= rangeMin && timeInSeconds < rangeMax;
  }

  /**
   * Returns true iff the given time in seconds is approximately (+/- 75%) in the range of valid
   * execution times for this TestSize.
   */
  public boolean isInRangeFuzzy(Integer timeInSeconds) {
    return timeInSeconds >= rangeMin - (rangeMin * .75)
        && (this == ETERNAL || timeInSeconds <= rangeMax + (rangeMax * .75));
  }

  /**
   * Returns suggested test size for the given time in seconds.
   */
  public static TestTimeout getSuggestedTestTimeout(Integer timeInSeconds) {
    for (TestTimeout testTimeout : values()) {
      if (testTimeout.isInRangeExact(timeInSeconds)) {
        return testTimeout;
      }
    }
    return ETERNAL;
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
      return TestTimeout.valueOf(attr.toUpperCase());
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  /**
   * Converter for the --test_timeout option.
   */
  public static class TestTimeoutConverter implements Converter<Map<TestTimeout, Integer>> {
    public TestTimeoutConverter() {}

    @Override
    public Map<TestTimeout, Integer> convert(String input) throws OptionsParsingException {
      List<Integer> values = new ArrayList<>();
      for (String token : Splitter.on(',').limit(6).split(input)) {
        // Handle the case of "2," which is accepted as legal... Because Splitter.split is lazy,
        // there's no way of knowing if an empty string is a trailing or an intermediate one,
        // so we can't fully emulate String.split(String, 0).
        if (!token.isEmpty() || values.size() > 1) {
          try {
            values.add(Integer.valueOf(token));
          } catch (NumberFormatException e) {
            throw new OptionsParsingException("'" + input + "' is not an int");
          }
        }
      }
      EnumMap<TestTimeout, Integer> timeouts = Maps.newEnumMap(TestTimeout.class);
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
        if (!timeouts.containsKey(label) || timeouts.get(label) <= 0) {
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
