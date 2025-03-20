// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import java.util.Objects;

/**
 * A filtering policy that excludes multiple single targets. These are not expected to be a part of
 * any SkyKey and it's expected that the number of targets is not too large.
 */
class TargetExcludingFilteringPolicy implements FilteringPolicy {
  private final ImmutableSet<Label> excludedSingleTargets;

  TargetExcludingFilteringPolicy(ImmutableSet<Label> excludedSingleTargets) {
    this.excludedSingleTargets = excludedSingleTargets;
  }

  @Override
  public boolean shouldRetain(Target target, boolean explicit) {
    return !excludedSingleTargets.contains(target.getLabel());
  }

  @Override
  public String toString() {
    return String.format("excludedTargets%s", excludedSingleTargets);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof TargetExcludingFilteringPolicy that)) {
      return false;
    }
    return Objects.equals(excludedSingleTargets, that.excludedSingleTargets);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(excludedSingleTargets);
  }
}
