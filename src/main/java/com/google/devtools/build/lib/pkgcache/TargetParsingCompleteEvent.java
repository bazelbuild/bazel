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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import java.util.Collection;
import java.util.List;

/** This event is fired just after target pattern evaluation is completed. */
public class TargetParsingCompleteEvent implements BuildEvent {

  private final ImmutableList<String> originalTargetPattern;
  private final ImmutableSet<Target> targets;
  private final ImmutableSet<Target> filteredTargets;
  private final ImmutableSet<Target> testFilteredTargets;
  private final ImmutableSet<Target> expandedTargets;

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
      Collection<Target> expandedTargets) {
    this.targets = ImmutableSet.copyOf(targets);
    this.filteredTargets = ImmutableSet.copyOf(filteredTargets);
    this.testFilteredTargets = ImmutableSet.copyOf(testFilteredTargets);
    this.originalTargetPattern = ImmutableList.copyOf(originalTargetPattern);
    this.expandedTargets = ImmutableSet.copyOf(expandedTargets);
  }

  @VisibleForTesting
  public TargetParsingCompleteEvent(Collection<Target> targets) {
    this(
        targets,
        ImmutableSet.<Target>of(),
        ImmutableSet.<Target>of(),
        ImmutableList.<String>of(),
        targets);
  }

  /**
   * @return the parsed targets, which will subsequently be loaded
   */
  public ImmutableSet<Target> getTargets() {
    return targets;
  }

  public Iterable<Label> getLabels() {
    return Iterables.transform(targets, Target::getLabel);
  }

  /**
   * @return the filtered targets (i.e., using -//foo:bar on the command-line)
   */
  public ImmutableSet<Target> getFilteredTargets() {
    return filteredTargets;
  }

  /**
   * @return the test-filtered targets, if --build_test_only is in effect
   */
  public ImmutableSet<Target> getTestFilteredTargets() {
    return testFilteredTargets;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.targetPatternExpanded(originalTargetPattern);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder<BuildEventId> childrenBuilder = ImmutableList.builder();
    for (Target target : expandedTargets) {
      // Test suits won't produce target configuration and  target-complete events, so do not
      // announce here completion as children.
      if (!TargetUtils.isTestSuiteRule(target)) {
        childrenBuilder.add(BuildEventId.targetConfigured(target.getLabel()));
      }
    }
    return childrenBuilder.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    return GenericBuildEvent.protoChaining(this)
        .setExpanded(BuildEventStreamProtos.PatternExpanded.newBuilder().build())
        .build();
  }
}
