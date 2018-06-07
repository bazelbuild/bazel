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
import java.util.List;
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

  /**
   * Construct the event.
   *
   * @param targets The targets that were parsed from the command-line pattern.
   */
  public TargetParsingCompleteEvent(
      Collection<Target> targets,
      Collection<Target> filteredTargets,
      Collection<Target> testFilteredTargets,
      List<String> originalTargetPattern,
      Collection<Target> expandedTargets,
      List<String> failedTargetPatterns) {
    this.targets = asThinTargets(targets);
    this.filteredTargets = asThinTargets(filteredTargets);
    this.testFilteredTargets = asThinTargets(testFilteredTargets);
    this.originalTargetPattern = ImmutableList.copyOf(originalTargetPattern);
    this.expandedTargets = asThinTargets(expandedTargets);
    this.failedTargetPatterns = ImmutableList.copyOf(failedTargetPatterns);
  }

  @VisibleForTesting
  public TargetParsingCompleteEvent(Collection<Target> targets) {
    this(
        targets,
        ImmutableSet.<Target>of(),
        ImmutableSet.<Target>of(),
        ImmutableList.<String>of(),
        targets,
        ImmutableList.<String>of());
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

  /** @return the filtered targets (i.e., using -//foo:bar on the command-line) */
  public ImmutableSet<ThinTarget> getFilteredTargets() {
    return filteredTargets;
  }

  /** @return the test-filtered targets, if --build_test_only is in effect */
  public ImmutableSet<ThinTarget> getTestFilteredTargets() {
    return testFilteredTargets;
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
