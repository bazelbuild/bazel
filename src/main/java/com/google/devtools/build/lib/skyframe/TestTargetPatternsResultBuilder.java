// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicates;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TestTargetUtils;
import com.google.devtools.build.lib.pkgcache.TargetProvider;
import java.util.ArrayList;

/**
 * Evaluates set of test targets based on list of target patterns. In contradistinction to 
 * {@code BuildTargetPatternEvaluatorUtil} this class will expand all test suites. 
 */
class TestTargetPatternsResultBuilder extends TargetPatternsResultBuilder {
  private final ArrayList<ResolvedTargetsOfPattern> labelsOfPatterns = new ArrayList<>(); 
  private final TargetProvider targetProvider;
  private final ExtendedEventHandler eventHandler;
  private final boolean keepGoing;

  private static class ResolvedTargetsOfPattern {
    private final ResolvedTargets<Label> labels;
    private final boolean negativePattern;

    public ResolvedTargetsOfPattern(ResolvedTargets<Label> labels, boolean negativePattern) {
      this.labels = labels;
      this.negativePattern = negativePattern;
    }

    public ResolvedTargets<Label> getLabels() {
      return labels;
    }

    public boolean isNegativePattern() {
      return negativePattern;
    }
  }

  TestTargetPatternsResultBuilder(
      TargetProvider targetProvider, ExtendedEventHandler eventHandler, boolean keepGoing) {
    this.targetProvider = targetProvider;
    this.eventHandler = eventHandler;
    this.keepGoing = keepGoing;
  }

  @Override
  void addLabelsOfNegativePattern(ResolvedTargets<Label> labels) {
    labelsOfPatterns.add(new ResolvedTargetsOfPattern(labels, true));
  }

  @Override
  void addLabelsOfPositivePattern(ResolvedTargets<Label> labels) {
    labelsOfPatterns.add(new ResolvedTargetsOfPattern(labels, false));
  }

  @Override
  protected Iterable<Label> getLabels() {
    ArrayList<Label> labels = new ArrayList<>();
    for (ResolvedTargetsOfPattern resolvedLabels : labelsOfPatterns) {
      labels.addAll(resolvedLabels.getLabels().getTargets());
      labels.addAll(resolvedLabels.getLabels().getFilteredTargets());
    }
    return labels;
  }

  @Override
  ResolvedTargets.Builder<Target> buildInternal() throws TargetParsingException {
    ResolvedTargets.Builder<Target> finalResult = ResolvedTargets.builder();
    for (ResolvedTargetsOfPattern labels : labelsOfPatterns) {
      ResolvedTargets.Builder<Target> resolvedTargetsBuilder =
          transformLabelsIntoTargets(labels.getLabels());
      ResolvedTargets<Target> expanded = TestTargetUtils.expandTestSuites(targetProvider,
          eventHandler,
          resolvedTargetsBuilder.build().getTargets(),
          /*strict=*/false,
          keepGoing);
      if (expanded.hasError()) {
        setError();
      }
      if (labels.isNegativePattern()) {
        finalResult.filter(Predicates.not(Predicates.in(expanded.getTargets())));
      } else {
        finalResult.addAll(expanded.getTargets());
      }
    }
    return finalResult;
  }
}
