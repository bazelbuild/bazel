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

import com.google.devtools.build.lib.util.Pair;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Utility functions over test Targets that don't really belong in the base {@link Target}
 * interface.
 */
public final class TestTargetUtils {

  /**
   * Returns whether a test with the specified tags matches a filter (as specified by the set
   * of its positive and its negative filters).
   */
  public static boolean testMatchesFilters(
      Collection<String> testTags,
      Collection<String> requiredTags,
      Collection<String> excludedTags,
      boolean mustMatchAllPositive) {
    for (String tag : excludedTags) {
      if (testTags.contains(tag)) {
        return false;
      }
    }

    // Check required tags, if there are any.
    if (requiredTags.isEmpty()) {
      return true;
    } else if (mustMatchAllPositive) {
      // Require all tags to be present.
      for (String tag : requiredTags) {
        if (!testTags.contains(tag)) {
          return false;
        }
      }
      return true;
    } else {
      // Require at least one positive tag. If the two collections are not disjoint, then they have
      // at least one element in common.
      return !Collections.disjoint(requiredTags, testTags);
    }
  }

  /**
   * Decides whether to include a test in a test_suite or not.
   * @param testTags Collection of all tags exhibited by a given test.
   * @param requiredTags Tags declared by the suite. A Test must match ALL of these.
   * @param excludedTags Tags declared by the suite. A Test must match NONE of these.
   * @return false is the test is to be removed.
   */
  public static boolean testMatchesFilters(
      Collection<String> testTags,
      Collection<String> requiredTags,
      Collection<String> excludedTags) {
    return testMatchesFilters(
        testTags, requiredTags, excludedTags, /* mustMatchAllPositive= */ true);
  }

  /**
   * Decides whether to include a test in a test_suite or not.
   * @param testTarget A given test target.
   * @param requiredTags Tags declared by the suite. A Test must match ALL of these.
   * @param excludedTags Tags declared by the suite. A Test must match NONE of these.
   * @return false is the test is to be removed.
   */
  private static boolean testMatchesFilters(
      Rule testTarget,
      Collection<String> requiredTags,
      Collection<String> excludedTags) {
    AttributeMap nonConfigurableAttrs = NonconfigurableAttributeMapper.of(testTarget);
    Set<String> testTags = new HashSet<>(nonConfigurableAttrs.get("tags", Types.STRING_LIST));
    testTags.add(nonConfigurableAttrs.get("size", Type.STRING));
    return testMatchesFilters(testTags, requiredTags, excludedTags);
  }

  /**
   * Filters 'tests' (by mutation) according to the 'tags' attribute, specifically those that
   * match ALL of the tags in tagsAttribute.
   *
   * @precondition {@code env.getAccessor().isTestSuite(testSuite)}
   * @precondition {@code env.getAccessor().isTestRule(test)} for all test in tests
   */
  public static void filterTests(Rule testSuite, Set<Target> tests) {
    List<String> tagsAttribute =
        NonconfigurableAttributeMapper.of(testSuite).get("tags", Types.STRING_LIST);
    // Split the tags list into positive and negative tags
    Pair<Collection<String>, Collection<String>> tagLists = sortTagsBySense(tagsAttribute);
    Collection<String> positiveTags = tagLists.first;
    Collection<String> negativeTags = tagLists.second;
    tests.removeIf((Target t) -> !testMatchesFilters((Rule) t, positiveTags, negativeTags));
  }

  /**
   * Separates a list of text "tags" into a Pair of Collections, where
   * the first element are the required or positive tags and the second element
   * are the excluded or negative tags.
   * This should work on tag list provided from the command line
   * --test_tags_filters flag or on tag filters explicitly declared in the
   * suite.
   *
   * @param tagList A collection of text targets to separate.
   */
  public static Pair<Collection<String>, Collection<String>> sortTagsBySense(
      Iterable<String> tagList) {
    Collection<String> requiredTags = new HashSet<>();
    Collection<String> excludedTags = new HashSet<>();

    for (String tag : tagList) {
      if (tag.startsWith("-")) {
        excludedTags.add(tag.substring(1));
      } else if (tag.startsWith("+")) {
        requiredTags.add(tag.substring(1));
      } else if (!tag.equals("manual")) {
        // Ignore manual attribute because it is an exception: it is not a filter but a property of
        // test_suite.
        requiredTags.add(tag);
      }
    }
    return Pair.of(requiredTags, excludedTags);
  }
}
