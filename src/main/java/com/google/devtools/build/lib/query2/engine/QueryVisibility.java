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

package com.google.devtools.build.lib.query2.engine;

import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;

/**
 * A predicate for targets included in the visibility list of a rule.
 *
 * <p>A rule's visibility is described by a set of {@link QueryVisibility}. Each element
 * in the set corresponds to an entry in the rule's visibility attribute, or an entry in the
 * packages attribute of an included package_group.
 */
public abstract class QueryVisibility<T> {

  /** Returns true if the visibility specification includes the given target's package. */
  public abstract boolean contains(T target);

  /** Global visibility. */
  @SuppressWarnings("unchecked")  // Safe covariant cast.
  public static <T> QueryVisibility<T> everything() {
    return (QueryVisibility<T>) (Object) EVERYTHING;
  }

  private static final QueryVisibility<?> EVERYTHING = new QueryVisibility<Object>() {
    @Override
    public boolean contains(Object target) {
      return true;
    }

    @Override
    public String toString() {
      return "QueryVisibility(//visibility:public)";
    }
  };

  /**
   * Same-package visibility.
   *
   * <p>Targets are always visible to other targets in the same package. Additionally, targets
   * under java/ are always visible to the corresponding package in javatests/. 
   */
  public static <T> QueryVisibility<T> samePackage(T from, TargetAccessor<T> accessor) {
    return new SamePackageVisibility<>(from, accessor);
  }

  private static final class SamePackageVisibility<T> extends QueryVisibility<T> {

    private static final String JAVA_PREFIX = "java/";
    private static final String JAVATESTS_PREFIX = "javatests/";

    private final String packageName;
    private final TargetAccessor<T> accessor;

    public SamePackageVisibility(T target, TargetAccessor<T> accessor) {
      this.packageName = accessor.getPackage(target);
      this.accessor = accessor;
    }

    @Override
    public boolean contains(T target) {
      String other = accessor.getPackage(target);
      if (packageName.equals(other)) {
        return true;
      }

      // packages in java/ are always visible from the corresponding package in javatests/
      if (other.startsWith(JAVATESTS_PREFIX)
          && packageName.equals(JAVA_PREFIX + other.substring(JAVATESTS_PREFIX.length()))) {
        return true;
      }

      return false;
    }

    @Override
    public String toString() {
      return String.format("QueryVisibility(samePackage=%s)", "<PACKAGE>");
    }
  }
}
