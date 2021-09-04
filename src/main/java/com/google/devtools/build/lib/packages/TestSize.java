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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Set;

/**
 * Possible test sizes.
 *
 * Test size may affect the way how test is executed - e.g., it will determine
 * default timeout value and estimated local resource usage.
 */
public enum TestSize {

  // Small tests use small amount of memory, but CPU intensive.
  SMALL(TestTimeout.SHORT, 2),
  // Medium tests tend to use larger amount of memory.
  MEDIUM(TestTimeout.MODERATE, 10),
  // All other tests estimated to use fairly large amount of memory.
  LARGE(TestTimeout.LONG, 20),
  ENORMOUS(TestTimeout.ETERNAL, 30);

  // Memoize canonical lowercase name -> TestSize mappings to avoid extraneous toUpperCases for
  // valueOf.
  private static final ImmutableMap<String, TestSize> CANONICAL_LOWER_CASE_NAME_TABLE;
  static {
    ImmutableMap.Builder<String, TestSize> builder = ImmutableMap.builder();
    for (TestSize size : TestSize.values()) {
      builder.put(size.name().toLowerCase(), size);
    }
    CANONICAL_LOWER_CASE_NAME_TABLE = builder.build();
  }

  private final TestTimeout timeout;
  private final int defaultShards;

  TestSize(TestTimeout defaultTimeout, int defaultShards) {
    this.timeout = defaultTimeout;
    this.defaultShards = defaultShards;
  }

  /**
   * Returns default timeout in seconds.
   */
  public TestTimeout getDefaultTimeout() {
    return timeout;
  }

  /**
   * Returns default number of shards.
   */
  public int getDefaultShards() { return defaultShards; }

  /**
   * Returns test size of the given test target, or null if the size attribute is unrecognized.
   */
  public static TestSize getTestSize(Rule testTarget) {
    String attr = NonconfigurableAttributeMapper.of(testTarget).get("size", Type.STRING);
    return getTestSize(attr);
  }

  /**
   * Returns {@link TestSize} matching the given timeout or null if the
   * given timeout doesn't match any {@link TestSize}.
   *
   * @param timeout The timeout associated with the desired TestSize.
   */
  public static TestSize getTestSize(TestTimeout timeout) {
    for (TestSize size : TestSize.values()) {
      if (size.timeout == timeout) {
        return size;
      }
    }
    return null;
  }

  /**
   * Normal practice is to always use size tags as lower case strings.
   */
  @Override
  public String toString() {
    return super.toString().toLowerCase();
  }

  /**
   * Returns the enum associated with a test's size or null if the tag is
   * not lower case or an unknown size.
   */
  public static TestSize getTestSize(String attr) {
    if (!attr.equals(attr.toLowerCase())) {
      return null;
    }
    try {
      return CANONICAL_LOWER_CASE_NAME_TABLE.get(attr);
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  /**
   * Converter for the --test_size_filters option.
   */
  public static class TestSizeFilterConverter extends EnumFilterConverter<TestSize> {
    public TestSizeFilterConverter() {
      super(TestSize.class, "test size");
    }

    /**
     * {@inheritDoc}
     *
     * <p>This override is necessary to prevent OptionsData
     * from throwing a "must be assignable from the converter return type" exception.
     * OptionsData doesn't recognize the generic type and actual type are the same.
     */
    @Override
    public final Set<TestSize> convert(String input) throws OptionsParsingException {
      return super.convert(input);
    }
  }
}
