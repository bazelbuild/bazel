// Copyright 2011 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Simple options parser for JUnit 4.
 *
 * <p>
 * For the options "test_filter" and "test_exclude_filter", this class properly handles arguments in
 * either the form "--test_filter=foo" or "--test_filter foo".
 */
class JUnit4Options {

  public static final String TEST_INCLUDE_FILTER_OPTION = "--test_filter";
  public static final String TEST_EXCLUDE_FILTER_OPTION = "--test_exclude_filter";

  // This gets passed in by the build system.
  private static final String TESTBRIDGE_TEST_ONLY = "TESTBRIDGE_TEST_ONLY";

  /**
   * Parses the given array of arguments and returns a JUnit4Options
   * object representing the parsed arguments.
   */
  static JUnit4Options parse(Map<String, String> envVars, List<String> args) {
    ImmutableList.Builder<String> unparsedArgsBuilder = ImmutableList.builder();
    Map<String, String> optionsMap = Maps.newHashMap();

    optionsMap.put(TEST_INCLUDE_FILTER_OPTION, null);
    optionsMap.put(TEST_EXCLUDE_FILTER_OPTION, null);

    for (Iterator<String> it = args.iterator(); it.hasNext();) {
      String arg = it.next();
      int indexOfEquals = arg.indexOf("=");

      if (indexOfEquals > 0) {
        String optionName = arg.substring(0, indexOfEquals);
        if (optionsMap.containsKey(optionName)) {
          optionsMap.put(optionName, arg.substring(indexOfEquals + 1));
          continue;
        }
      } else if (optionsMap.containsKey(arg)) {
        // next argument is the regexp
        if (!it.hasNext()) {
          throw new RuntimeException("No filter expression specified after " + arg);
        }
        optionsMap.put(arg, it.next());
        continue;
      }
      unparsedArgsBuilder.add(arg);
    }
    // If TESTBRIDGE_TEST_ONLY is set in the environment, forward it to the
    // --test_filter flag.
    String testFilter = envVars.get(TESTBRIDGE_TEST_ONLY);
    if (testFilter != null && optionsMap.get(TEST_INCLUDE_FILTER_OPTION) == null) {
      optionsMap.put(TEST_INCLUDE_FILTER_OPTION, testFilter);
    }

    ImmutableList<String> unparsedArgs = unparsedArgsBuilder.build();
    return new JUnit4Options(optionsMap.get(TEST_INCLUDE_FILTER_OPTION),
                             optionsMap.get(TEST_EXCLUDE_FILTER_OPTION),
                             unparsedArgs.toArray(new String[unparsedArgs.size()]));
  }

  private final String testIncludeFilter;
  private final String testExcludeFilter;
  private final String[] unparsedArgs;

  @VisibleForTesting
  JUnit4Options(@Nullable String testIncludeFilter, @Nullable String testExcludeFilter,
                String[] unparsedArgs) {
    this.testIncludeFilter = testIncludeFilter;
    this.testExcludeFilter = testExcludeFilter;
    this.unparsedArgs = unparsedArgs;
  }

  /**
   * Returns the value of the test_filter option, or <code>null</code> if
   * it was not specified.
   */
  String getTestIncludeFilter() {
    return testIncludeFilter;
  }

  /**
   * Returns the value of the test_exclude_filter option, or <code>null</code> if
   * it was not specified.
   */
  String getTestExcludeFilter() {
    return testExcludeFilter;
  }

  /**
   * Returns an array of the arguments that did not match any known option.
   */
  String[] getUnparsedArgs() {
    return unparsedArgs;
  }
}
