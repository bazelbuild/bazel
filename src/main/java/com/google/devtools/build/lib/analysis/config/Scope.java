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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/**
 * Scope of a {@link BuildOptions} is defined by the {@link Scope.ScopeType} and {@link
 * Scope.ScopeDefinition}.
 */
public class Scope {
  public static final String CUSTOM_EXEC_SCOPE_PREFIX = "exec:--";

  /** Type of supported scopes. */
  @AutoCodec
  public record ScopeType(String scopeType) {
    /** The flag's value never changes except explicitly by a configuration transition. */
    public static final String UNIVERSAL = "universal";

    /** The flag's value resets on exec transitions. */
    public static final String TARGET = "target";

    /** The flag resets on targets outside the flag's project. See PROJECT.scl. */
    public static final String PROJECT = "project";

    /** Placeholder for flags that don't explicitly specify scope. Shouldn't be set directly. */
    public static final String DEFAULT = "default";

    public ScopeType {
      if (!(scopeType.equals(DEFAULT)
          || scopeType.equals(UNIVERSAL)
          || scopeType.equals(TARGET)
          || scopeType.equals(PROJECT)
          || scopeType.startsWith("exec:"))) {
        // TODO: don't let blaze crash for an invalid scope type.
        throw new IllegalArgumentException("Invalid scope type: " + scopeType);
      }
    }

    /** Which values can a rule's {@code scope} attribute have? */
    public static ImmutableList<String> allowedAttributeValues() {
      return ImmutableList.of(UNIVERSAL, TARGET, PROJECT);
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
