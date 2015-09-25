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
import com.google.common.base.Function;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;

import java.util.Collection;

/**
 * This event is fired just after target pattern evaluation is completed.
 */
public class TargetParsingCompleteEvent {

  private final ImmutableSet<Target> targets;
  private final ImmutableSet<Target> filteredTargets;
  private final ImmutableSet<Target> testFilteredTargets;
  private final long timeInMs;

  /**
   * Construct the event.
   * @param targets The targets that were parsed from the
   *     command-line pattern.
   */
  public TargetParsingCompleteEvent(Collection<Target> targets,
      Collection<Target> filteredTargets, Collection<Target> testFilteredTargets,
      long timeInMs) {
    this.timeInMs = timeInMs;
    this.targets = ImmutableSet.copyOf(targets);
    this.filteredTargets = ImmutableSet.copyOf(filteredTargets);
    this.testFilteredTargets = ImmutableSet.copyOf(testFilteredTargets);
  }

  @VisibleForTesting
  public TargetParsingCompleteEvent(Collection<Target> targets) {
    this(targets, ImmutableSet.<Target>of(), ImmutableSet.<Target>of(), 0);
  }

  /**
   * @return the parsed targets, which will subsequently be loaded
   */
  public ImmutableSet<Target> getTargets() {
    return targets;
  }

  public Iterable<Label> getLabels() {
    return Iterables.transform(targets, new Function<Target, Label>() {
      @Override
      public Label apply(Target input) {
        return input.getLabel();
      }
    });
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

  public long getTimeInMs() {
    return timeInMs;
  }
}
