// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.coverageoutputgenerator;

import static com.google.common.base.Verify.verify;

import com.google.auto.value.AutoValue;

/**
 * Stores MC/DC (Modified Condition/Decision Coverage) information.
 *
 * <p>Corresponds to MCDC lines in an lcov report with the form:
 *
 * <pre>MCDC:[line_number],[group_size],[sense],[taken],[index],[expression]</pre>
 *
 * where:
 * <ul>
 *   <li>line_number = the line number where the condition appears
 *   <li>group_size = the total number of conditions in this MC/DC group
 *   <li>sense = 't' or 'f' indicating whether this condition sensitizes for true or false
 *   <li>taken = number of times this condition was sensitized (0 if never hit, "-" parsed as 0)
 *   <li>index = unique index within the MC/DC group for this condition
 *   <li>expression = textual representation of the Boolean expression (optional)
 * </ul>
 */
@AutoValue
abstract class McdcCoverage {

  /**
   * Creates an instance of McdcCoverage.
   *
   * @param lineNumber The line number where the condition appears
   * @param groupSize The total number of conditions in this MC/DC group
   * @param sense The sense character ('t' or 'f') for this condition
   * @param taken Number of times this condition was sensitized
   * @param index Unique index within the MC/DC group
   * @param expression Textual representation of the Boolean expression
   */
  static McdcCoverage create(
      int lineNumber,
      int groupSize,
      char sense,
      long taken,
      int index,
      String expression) {
    return new AutoValue_McdcCoverage(
        lineNumber, groupSize, sense, taken, index, expression);
  }

  /**
   * Checks if two MC/DC records have the same identifying characteristics (can be merged).
   *
   * <p>Two MC/DC records can be merged if they have the same lineNumber, groupSize, sense,
   * index, and expression.
   *
   * @param first The first MC/DC record
   * @param second The second MC/DC record
   * @return true if the records can be merged, false otherwise
   */
  static boolean canMerge(McdcCoverage first, McdcCoverage second) {
    return first.lineNumber() == second.lineNumber()
        && first.groupSize() == second.groupSize()
        && first.sense() == second.sense()
        && first.index() == second.index()
        && first.expression().equals(second.expression());
  }

  /**
   * Merges two given instances of {@link McdcCoverage}.
   *
   * <p>Calling {@code lineNumber()}, {@code groupSize()}, {@code sense()}, {@code index()},
   * and {@code expression()} must return the same values for {@code first} and {@code second}.
   */
  static McdcCoverage merge(McdcCoverage first, McdcCoverage second) {
    verify(McdcCoverage.canMerge(first, second), "MC/DC records must match");

    return create(
        first.lineNumber(),
        first.groupSize(),
        first.sense(),
        first.taken() + second.taken(),
        first.index(),
        first.expression());
  }

  abstract int lineNumber();

  abstract int groupSize();

  abstract char sense();

  abstract long taken();

  abstract int index();

  abstract String expression();

  boolean wasHit() {
    return taken() > 0;
  }
}
