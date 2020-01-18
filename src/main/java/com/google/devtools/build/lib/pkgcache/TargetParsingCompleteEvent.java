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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import java.util.Collection;
import javax.annotation.Nullable;

/** This event is fired just after target pattern evaluation is completed. */
public class TargetParsingCompleteEvent implements BuildEventWithOrderConstraint {
  /** A target-like object that is lighter than a target but has all data needed by callers. */
  public static class ThinTarget {
    private final Label label;
    @Nullable private final String ruleClass;
    private final String targetKind;

    private ThinTarget(Target target) {
      this.label = target.getLabel();
      this.targetKind = target.getTargetKind();
      this.ruleClass = (target instanceof Rule) ? ((Rule) target).getRuleClass() : null;
    }

    public boolean isRule() {
      return ruleClass != null;
    }

    public String getTargetKind() {
      return targetKind;
    }

    public Label getLabel() {
      return label;
    }

    /** Gets the rule class of this target. Caller must already know it {@link #isRule}. */
    public String getRuleClass() {
      return Preconditions.checkNotNull(ruleClass, label);
    }

    public boolean isTestSuiteRule() {
      return isRule() && TargetUtils.isTestSuiteRuleName(getRuleClass());
    }

    public boolean isNotATestOrTestSuite() {
      return !isRule() || (!isTestSuiteRule() && !TargetUtils.isTestRuleName(getRuleClass()));
    }
  }

  private final ImmutableList<String> originalTargetPattern;
  private final ImmutableList<String> failedTargetPatterns;
  private final ImmutableSet<ThinTarget> targets;
  private final ImmutableSet<ThinTarget> filteredTargets;
  private final ImmutableSet<ThinTarget> testFilteredTargets;
  private final ImmutableSet<ThinTarget> expandedTargets;
  private final ImmutableSetMultimap<String, Label> originalPatternsToLabels;

  /**
   * Construct the event.
   *
   * @param targets The targets that were parsed from the command-line pattern.
   */
  public TargetParsingCompleteEvent(
      Collection<Target> targets,
      Collection<Target> filteredTargets,
      Collection<Target> testFilteredTargets,
      ImmutableList<String> originalTargetPattern,
      Collection<Target> expandedTargets,
      ImmutableList<String> failedTargetPatterns,
      ImmutableSetMultimap<String, Label> originalPatternsToLabels) {
    this.targets = asThinTargets(targets);
    this.filteredTargets = asThinTargets(filteredTargets);
    this.testFilteredTargets = asThinTargets(testFilteredTargets);
    this.originalTargetPattern = Preconditions.checkNotNull(originalTargetPattern);
    this.expandedTargets = asThinTargets(expandedTargets);
    this.failedTargetPatterns = Preconditions.checkNotNull(failedTargetPatterns);
    this.originalPatternsToLabels = originalPatternsToLabels;
  }

  @VisibleForTesting
  public TargetParsingCompleteEvent(Collection<Target> targets) {
    this(
        targets,
        ImmutableSet.of(),
        ImmutableSet.of(),
        ImmutableList.of(),
        targets,
        ImmutableList.of(),
        ImmutableSetMultimap.of());
  }

  public ImmutableList<String> getOriginalTargetPattern() {
    return originalTargetPattern;
  }

  public ImmutableList<String> getFailedTargetPatterns() {
    return failedTargetPatterns;
  }

  /** @return the parsed targets, which will subsequently be loaded */
  public ImmutableSet<ThinTarget> getTargets() {
    return targets;
  }

  public Iterable<Label> getLabels() {
    return Iterables.transform(targets, ThinTarget::getLabel);
  }

  public Iterable<Label> getFilteredLabels() {
    return Iterables.transform(filteredTargets, ThinTarget::getLabel);
  }

  public Iterable<Label> getTestFilteredLabels() {
    return Iterables.transform(testFilteredTargets, ThinTarget::getLabel);
  }

  /** @return the filtered targets (i.e., using -//foo:bar on the command-line) */
  public ImmutableSet<ThinTarget> getFilteredTargets() {
    return filteredTargets;
  }

  /** @return the test-filtered targets, if --build_test_only is in effect */
  public ImmutableSet<ThinTarget> getTestFilteredTargets() {
    return testFilteredTargets;
  }

  /**
   * Returns a mapping from patterns originally passed on the command line to the labels they were
   * expanded to.
   *
   * <p>Negative patterns are not included here. Neither are labels of targets that are skipped due
   * to matching a negative pattern (even if they also matched a positive pattern).
   *
   * <p>Test suite labels are included here, but not the labels of the tests that the suite expanded
   * to.
   */
  public ImmutableSetMultimap<String, Label> getOriginalPatternsToLabels() {
    return originalPatternsToLabels;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.targetPatternExpanded(originalTargetPattern);
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.<BuildEventId>of(BuildEventId.buildStartedId());
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder<BuildEventId> childrenBuilder = ImmutableList.builder();
    for (String failedTargetPattern : failedTargetPatterns) {
      childrenBuilder.add(
          BuildEventId.targetPatternExpanded(ImmutableList.of(failedTargetPattern)));
    }
    for (ThinTarget target : expandedTargets) {
      // Test suits won't produce target configuration and  target-complete events, so do not
      // announce here completion as children.
      if (!target.isTestSuiteRule()) {
        childrenBuilder.add(BuildEventId.targetConfigured(target.getLabel()));
      }
    }
    return childrenBuilder.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this)
        .setExpanded(BuildEventStreamProtos.PatternExpanded.newBuilder().build())
        .build();
  }

  private static ImmutableSet<ThinTarget> asThinTargets(Collection<Target> targets) {
    return targets.stream().map(ThinTarget::new).collect(ImmutableSet.toImmutableSet());
  }
}
