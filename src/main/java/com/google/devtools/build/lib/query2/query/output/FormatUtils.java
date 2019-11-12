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

package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import java.util.Comparator;

/**
 * Given a set of query options, returns a BinaryPredicate suitable for passing to {@link
 * Rule#getLabels()}, {@link XmlOutputFormatter}, etc.
 */
class FormatUtils {

  private FormatUtils() {}

  static DependencyFilter getDependencyFilter(CommonQueryOptions queryOptions) {
    if (queryOptions.includeToolDeps) {
      return queryOptions.includeImplicitDeps
          ? DependencyFilter.ALL_DEPS
          : DependencyFilter.NO_IMPLICIT_DEPS;
    }
    return queryOptions.includeImplicitDeps
        ? DependencyFilter.ONLY_TARGET_DEPS
        : DependencyFilter.and(
            DependencyFilter.NO_IMPLICIT_DEPS, DependencyFilter.ONLY_TARGET_DEPS);
  }

  /** An ordering of Targets based on the ordering of their labels. */
  static class TargetOrdering implements Comparator<Target> {
    @Override
    public int compare(Target o1, Target o2) {
      return o1.getLabel().compareTo(o2.getLabel());
    }
  }

  /**
   * Returns the target location, eventually stripping out the workspace path to obtain a relative
   * target (stable across machines / workspaces).
   *
   * @param target The target to extract location from.
   * @param relative Whether to return a relative path or not.
   * @return the target location
   */
  static String getLocation(Target target, boolean relative) {
    Location location = target.getLocation();
    return relative
        ? location.print(
            target.getPackage().getPackageDirectory().asFragment(),
            target.getPackage().getNameFragment())
        : location.print();
  }
}
