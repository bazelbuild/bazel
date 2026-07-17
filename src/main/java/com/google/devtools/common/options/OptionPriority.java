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

import com.google.common.collect.ImmutableList;
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
  /**
   * Each option that is passed explicitly has 0 ancestors, so it only has its command line index
   * (or rc index, etc., depending on the category), but expanded options have the command line
   * index of its parent and then its position within the options that were expanded at that point.
   * Since options can expand to expanding options, and --config can expand to expansion options as
   * well, this can technically go arbitrarily deep, but in practice this is very short, of length <
   * 5, most commonly of length 1.
   */
  private final ImmutableList<Integer> priorityIndices;

  private boolean alreadyExpanded = false;

  private OptionPriority(
      PriorityCategory priorityCategory, ImmutableList<Integer> priorityIndices) {
    this.priorityCategory = priorityCategory;
    this.priorityIndices = priorityIndices;
  }

  /** Get the first OptionPriority for that category. */
  static OptionPriority lowestOptionPriorityAtCategory(PriorityCategory category) {
    return new OptionPriority(category, ImmutableList.of(0));
  }

  /**
   * Get the priority for the option following this one. In normal, incremental option parsing, the
   * returned priority would compareTo as after the current one. Does not increment ancestor
   * priorities.
   */
  static OptionPriority nextOptionPriority(OptionPriority priority) {
    int lastElementPosition = priority.priorityIndices.size() - 1;
    return new OptionPriority(
        priority.priorityCategory,
        ImmutableList.<Integer>builder()
            .addAll(priority.priorityIndices.subList(0, lastElementPosition))
            .add(priority.priorityIndices.get(lastElementPosition) + 1)
            .build());
  }

  /**
   * Some options are expanded to other options, and the children options need to have their order
   * preserved while maintaining their position between the options that flank the parent option.
   *
   * @return the priority for the first child of the passed priority. This child's ordering can be
   *     tracked the same way that the parent's was.
   */
  public static OptionPriority getChildPriority(OptionPriority parentPriority)
      throws OptionsParsingException {
    if (parentPriority.alreadyExpanded) {
      throw new OptionsParsingException("Tried to expand option too many times");
    }
    // Prevent this option from being re-expanded.
    parentPriority.alreadyExpanded = true;

    // The child priority has 1 more level of nesting than its parent.
    return new OptionPriority(
        parentPriority.priorityCategory,
        ImmutableList.<Integer>builder().addAll(parentPriority.priorityIndices).add(0).build());
  }

  public PriorityCategory getPriorityCategory() {
    return priorityCategory;
  }

  @Override
  public int compareTo(OptionPriority o) {
    if (priorityCategory.equals(o.priorityCategory)) {
      for (int i = 0; i < priorityIndices.size() && i < o.priorityIndices.size(); ++i) {
        if (!priorityIndices.get(i).equals(o.priorityIndices.get(i))) {
          return priorityIndices.get(i).compareTo(o.priorityIndices.get(i));
        }
      }
      // The values are up to the shorter one's length are the same, so the shorter one is a direct
      // ancestor and comes first.
      return Integer.compare(priorityIndices.size(), o.priorityIndices.size());
    }
    return Integer.compare(priorityCategory.ordinal(), o.priorityCategory.ordinal());
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof OptionPriority) {
      OptionPriority other = (OptionPriority) o;
      return priorityCategory.equals(other.priorityCategory)
          && priorityIndices.equals(other.priorityIndices);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(priorityCategory, priorityIndices);
  }

  @Override
  public String toString() {
    return String.format("OptionPriority(%s,%s)", priorityCategory, priorityIndices);
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
