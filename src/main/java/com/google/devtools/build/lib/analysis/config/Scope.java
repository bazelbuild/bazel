// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.Arrays.stream;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.Locale;
import javax.annotation.Nullable;

/**
 * Scope of a {@link BuildOptions} is defined by the {@link Scope.ScopeType} and {@link
 * Scope.ScopeDefinition}.
 */
public class Scope {
  /** Type of supported scopes. */
  public enum ScopeType {
    /** The flag's value never changes except explicitly by a configuraiton transition. * */
    UNIVERSAL,
    /** The flag's value resets on exec transitions. * */
    TARGET,
    /** The flag resets on targets outside the flag's project. See PROJECT.scl. * */
    PROJECT,
    /** Placeholder for flags that don't explicitly specify scope. Shouldn't be set directly. * */
    DEFAULT;

    /** Returns the enum of a {@code scope = "<string>"} value. */
    public static ScopeType valueOfIgnoreCase(String scopeType) throws IllegalArgumentException {
      return ScopeType.valueOf(scopeType.toUpperCase(Locale.ROOT));
    }

    /** Which values can a rule's {@code scope} attribute have? */
    public static ImmutableList<String> allowedAttributeValues() {
      return stream(ScopeType.values())
          .map(e -> e.name().toLowerCase(Locale.ROOT))
          .filter(e -> !e.equals("default")) // "default" is an internal value for unset attributes.
          .collect(toImmutableList());
    }
  }

  /**
   * Definition of a scope. Users can define this in their PROJECT.scl file that is in the same
   * directory as the BUILD file where the scoped flags are defined or in a parent directory. This
   * is only relevant if the scope type is PROJECT.
   */
  public static class ScopeDefinition {
    private final ImmutableSet<String> ownedCodePaths;

    public ScopeDefinition(ImmutableSet<String> ownedCodePaths) {
      this.ownedCodePaths = ownedCodePaths;
    }

    public ImmutableSet<String> getOwnedCodePaths() {
      return ownedCodePaths;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("ownedCodePaths", ownedCodePaths).toString();
    }
  }

  ScopeType scopeType;
  @Nullable ScopeDefinition scopeDefinition;

  public Scope(ScopeType scopeType, @Nullable ScopeDefinition scopeDefinition) {
    this.scopeType = scopeType;
    this.scopeDefinition = scopeDefinition;
  }

  public ScopeType getScopeType() {
    return scopeType;
  }

  @Nullable
  public ScopeDefinition getScopeDefinition() {
    return scopeDefinition;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("scopeType", scopeType)
        .add("scopeDefinition", scopeDefinition)
        .toString();
  }
}
