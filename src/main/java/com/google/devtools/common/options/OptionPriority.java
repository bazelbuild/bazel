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
package com.google.devtools.common.options;

import java.util.Objects;

/**
 * The position of an option in the interpretation order. Options are interpreted using a
 * last-option-wins system for single valued options, and are listed in that order for
 * multiple-valued options.
 *
 * <p>The position of the option is in category order, and within the priority category in index
 * order.
 */
public class OptionPriority implements Comparable<OptionPriority> {
  private final PriorityCategory priorityCategory;
  private final int index;

  private OptionPriority(PriorityCategory priorityCategory, int index) {
    this.priorityCategory = priorityCategory;
    this.index = index;
  }

  public static OptionPriority lowestOptionPriorityAtCategory(PriorityCategory category) {
    return new OptionPriority(category, 0);
  }

  public static OptionPriority nextOptionPriority(OptionPriority priority) {
    return new OptionPriority(priority.priorityCategory, priority.index + 1);
  }

  public PriorityCategory getPriorityCategory() {
    return priorityCategory;
  }

  @Override
  public int compareTo(OptionPriority o) {
    if (priorityCategory.equals(o.priorityCategory)) {
      return index - o.index;
    }
    return priorityCategory.ordinal() - o.priorityCategory.ordinal();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof OptionPriority) {
      OptionPriority other = (OptionPriority) o;
      return other.priorityCategory.equals(priorityCategory) && other.index == index;
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(priorityCategory, index);
  }

  /**
   * The priority of option values, in order of increasing priority.
   *
   * <p>In general, new values for options can only override values with a lower or equal priority.
   * Option values provided in annotations in an options class are implicitly at the priority {@code
   * DEFAULT}.
   *
   * <p>The ordering of the priorities is the source-code order. This is consistent with the
   * automatically generated {@code compareTo} method as specified by the Java Language
   * Specification. DO NOT change the source-code order of these values, or you will break code that
   * relies on the ordering.
   */
  public enum PriorityCategory {

    /**
     * The priority of values specified in the {@link Option} annotation. This should never be
     * specified in calls to {@link OptionsParser#parse}.
     */
    DEFAULT,

    /**
     * Overrides default options at runtime, while still allowing the values to be overridden
     * manually.
     */
    COMPUTED_DEFAULT,

    /** For options coming from a configuration file or rc file. */
    RC_FILE,

    /** For options coming from the command line. */
    COMMAND_LINE,

    /** For options coming from invocation policy. */
    INVOCATION_POLICY,

    /**
     * This priority can be used to unconditionally override any user-provided options. This should
     * be used rarely and with caution!
     */
    SOFTWARE_REQUIREMENT
  }
}
