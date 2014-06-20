// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view.test;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

import java.util.Set;

/**
 * This provider can be implemented by rules which need special environments to run in (especially
 * tests).
 */
@Immutable
public final class ExecutionRequirementProvider implements TransitiveInfoProvider {

  private final ImmutableSet<String> requirements;

  public ExecutionRequirementProvider(Set<String> requirements) {
    this.requirements = ImmutableSet.copyOf(requirements);
  }

  /**
   * Returns a set of string tags that indicate special execution requirements, such as hardware
   * platforms, web browsers, etc.
   */
  public ImmutableSet<String> getRequirements() {
    return requirements;
  }
}
