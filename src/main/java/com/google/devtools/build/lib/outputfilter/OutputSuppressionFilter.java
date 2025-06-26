// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.outputfilter;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventContext;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.OutputFilter;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** An {@link OutputFilter} that suppresses output based on a set of user-defined rules. */
public final class OutputSuppressionFilter implements OutputFilter {

  private final ImmutableList<SuppressionRule> rules;
  private final ImmutableList<AtomicInteger> matchCounts;
  private final AtomicInteger totalSuppressedCount = new AtomicInteger(0);

  public OutputSuppressionFilter(List<String> ruleStrings) {
    ImmutableList.Builder<SuppressionRule> rulesBuilder = ImmutableList.builder();
    for (String ruleString : ruleStrings) {
      rulesBuilder.add(SuppressionRule.create(ruleString));
    }
    this.rules = rulesBuilder.build();
    this.matchCounts =
        rules.stream().map(r -> new AtomicInteger(0)).collect(ImmutableList.toImmutableList());
  }

  @Override
  public boolean showOutput(Event event) {
    for (int i = 0; i < rules.size(); i++) {
      SuppressionRule rule = rules.get(i);
      if (matches(rule, event)) {
        matchCounts.get(i).incrementAndGet();
        totalSuppressedCount.incrementAndGet();
        return false;
      }
    }
    return true;
  }

  private boolean matches(SuppressionRule rule, Event event) {
    EventContext context = event.getProperty(EventContext.class);

    if (rule.hasKeyword("package")) {
      String pkg = context != null ? context.getPackage() : null;
      if (pkg == null && event.getLocation() != null) {
        pkg = Paths.get(event.getLocation().file()).getParent().toString();
      }
      if (pkg == null || !Pattern.matches(rule.getKeywordValue("package"), pkg)) {
        return false;
      }
    }

    if (rule.hasKeyword("target")) {
      String target = context != null ? context.getTargetLabel() : null;
      if (target == null || !Pattern.matches(rule.getKeywordValue("target"), target)) {
        return false;
      }
    }

    if (rule.hasKeyword("rule")) {
      String ruleClass = context != null ? context.getRuleClass() : null;
      if (ruleClass == null || !Pattern.matches(rule.getKeywordValue("rule"), ruleClass)) {
        return false;
      }
    }

    if (rule.hasKeyword("tag")) {
      String tag = event.getTag();
      if (tag == null || !Pattern.matches(rule.getKeywordValue("tag"), tag)) {
        return false;
      }
    }

    Matcher matcher = rule.getPattern().matcher(event.getMessage());
    return matcher.find();
  }

  /**
   * Verifies that all rules with an expected count matched exactly that many times.
   *
   * @param reporter the event handler to report warnings to
   */
  public void verifyCounts(EventHandler reporter) {
    for (int i = 0; i < rules.size(); i++) {
      SuppressionRule rule = rules.get(i);
      int expectedCount = rule.getExpectedCount();
      if (expectedCount != -1) {
        int actualCount = matchCounts.get(i).get();
        if (actualCount != expectedCount) {
          reporter.handle(
              Event.warn(
                  String.format(
                      "Suppression expectation violation for rule '%s': expected %d, got %d",
                      rule, expectedCount, actualCount)));
        }
      }
    }
  }

  /** Returns the total number of messages suppressed by this filter. */
  public int getTotalSuppressedCount() {
    return totalSuppressedCount.get();
  }
}