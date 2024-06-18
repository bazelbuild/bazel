// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Iterables;
import java.util.Locale;
import java.util.Set;
import net.starlark.java.spelling.SpellChecker;

/**
 * Some static utility functions for determining suggested targets when a user requests a
 * non-existent target.
 */
public final class TargetSuggester {
  private static final int MAX_SUGGESTED_TARGETS_SIZE = 10;

  private static final int MAX_SUGGESTION_EDIT_DISTANCE = 5;

  private TargetSuggester() {}

  /**
   * Given a nonexistent target and the targets in its package, suggest what the user may have
   * intended based on lexicographic closeness to the possibilities.
   *
   * <p>This will be pretty printed in the following forms:
   *
   * <p>No suggested targets -> "".
   *
   * <p>Suggested target "a" -> "a".
   *
   * <p>Suggested targets "a", "b" -> "a, or b"
   *
   * <p>Suggested targets "a", "b", "c" -> "a, b, or c".
   */
  static String suggestTargets(String input, Set<String> words) {
    ImmutableList<String> suggestedTargets = suggestedTargets(input, words);
    return prettyPrintTargets(suggestedTargets);
  }

  /**
   * Given a requested target and a Set of targets in the same package, return a list of the targets
   * closest to the requested target based on edit distance.
   *
   * <p>If any strings are identical minus capitalization changes, they will be returned. If any
   * other strings are exactly 1 character off, they will be returned. Otherwise, the 10 nearest
   * (within a small edit distance) will be returned.
   */
  @VisibleForTesting
  static ImmutableList<String> suggestedTargets(String input, Set<String> words) {

    final String lowerCaseInput = input.toLowerCase(Locale.US);

    // Add words based on edit distance
    ImmutableListMultimap.Builder<Integer, String> editDistancesBuilder =
        ImmutableListMultimap.builder();

    int maxEditDistance = Math.min(MAX_SUGGESTION_EDIT_DISTANCE, (input.length() + 1) / 2);
    for (String word : words) {
      String lowerCaseWord = word.toLowerCase(Locale.US);

      int editDistance = SpellChecker.editDistance(lowerCaseInput, lowerCaseWord, maxEditDistance);

      if (editDistance >= 0) {
        editDistancesBuilder.put(editDistance, word);
      }
    }
    ImmutableListMultimap<Integer, String> editDistanceToWords = editDistancesBuilder.build();

    ImmutableList<String> zeroEditDistanceWords = editDistanceToWords.get(0);
    ImmutableList<String> oneEditDistanceWords = editDistanceToWords.get(1);

    if (editDistanceToWords.isEmpty()) {
      return ImmutableList.of();
    } else if (!zeroEditDistanceWords.isEmpty()) {
      int sublistLength = Math.min(zeroEditDistanceWords.size(), MAX_SUGGESTED_TARGETS_SIZE);
      return ImmutableList.copyOf(zeroEditDistanceWords.subList(0, sublistLength));
    } else if (!oneEditDistanceWords.isEmpty()) {
      int sublistLength = Math.min(oneEditDistanceWords.size(), MAX_SUGGESTED_TARGETS_SIZE);
      return ImmutableList.copyOf(oneEditDistanceWords.subList(0, sublistLength));
    } else {
      return getSuggestedTargets(editDistanceToWords, maxEditDistance);
    }
  }

  /**
   * Given a map of edit distance values to words that are that distance from the requested target,
   * returns up to MAX_SUGGESTED_TARGETS_SIZE targets that are at least edit distance 2 but no more
   * than the given max away.
   */
  private static ImmutableList<String> getSuggestedTargets(
      ImmutableListMultimap<Integer, String> editDistanceToWords, int maxEditDistance) {
    // iterate through until MAX is achieved
    int total = 0;
    ImmutableList.Builder<String> suggestedTargets = ImmutableList.builder();
    for (int editDistance = 2;
        editDistance < maxEditDistance && total < MAX_SUGGESTED_TARGETS_SIZE;
        editDistance++) {

      ImmutableList<String> values = editDistanceToWords.get(editDistance);
      int addAmount = Math.min(values.size(), MAX_SUGGESTED_TARGETS_SIZE - total);
      suggestedTargets.addAll(values.subList(0, addAmount));
      total += addAmount;
    }

    return suggestedTargets.build();
  }

  /**
   * Create a pretty-printable String for a list. Joiner doesn't currently support multiple
   * separators so this is a custom roll for now. Returns a comma-delimited list with ", or " before
   * the last element.
   */
  @VisibleForTesting
  public static String prettyPrintTargets(ImmutableList<String> targets) {
    String targetString;
    if (targets.isEmpty()) {
      return "";
    } else if (targets.size() == 1) {
      targetString = targets.get(0);
    } else {
      String firstPart = Joiner.on(", ").join(targets.subList(0, targets.size() - 1));
      targetString = Joiner.on(", or ").join(firstPart, Iterables.getLast(targets));
    }
    return " (did you mean " + targetString + "?)";
  }
}
