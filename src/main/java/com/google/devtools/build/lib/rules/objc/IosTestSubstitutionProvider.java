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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides substitutions which should be performed to the test runner script of
 * {@code experimental_ios_test} targets.
 */
@Immutable
public final class IosTestSubstitutionProvider implements TransitiveInfoProvider {

  private final ImmutableList<Substitution> substitutions;

  public IosTestSubstitutionProvider(Iterable<Substitution> substitutions) {
    this.substitutions = ImmutableList.copyOf(substitutions);
  }

  /**
   * Returns a list of substitutions which should be performed to the test runner script.
   */
  public ImmutableList<Substitution> getSubstitutionsForTestRunnerScript() {
    return substitutions;
  }
}
