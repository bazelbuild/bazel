// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;

/**
 * Evaluates set of build targets based on list of target patterns.
 */
class BuildTargetPatternsResultBuilder extends TargetPatternsResultBuilder {
  private ResolvedTargets.Builder<Label> resolvedLabelsBuilder = ResolvedTargets.builder();

  @Override
  void addLabelsOfNegativePattern(ResolvedTargets<Label> labels) {
    resolvedLabelsBuilder.filter(Predicates.not(Predicates.in(labels.getTargets())));
  }

  @Override
  void addLabelsOfPositivePattern(ResolvedTargets<Label> labels) {
    resolvedLabelsBuilder.merge(labels);
  }

  @Override
  protected Iterable<Label> getLabels() {
    ResolvedTargets<Label> resolvedLabels = resolvedLabelsBuilder.build();
    return Iterables.concat(resolvedLabels.getTargets(), resolvedLabels.getFilteredTargets());
  }

  @Override
  ResolvedTargets.Builder<Target> buildInternal() throws TargetParsingException {
    return transformLabelsIntoTargets(resolvedLabelsBuilder.build());
  }
}
