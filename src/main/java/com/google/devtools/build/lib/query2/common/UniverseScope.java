// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.common;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import java.util.LinkedHashSet;
import java.util.Optional;

/** Representation of the --universe_scope option value. */
public abstract class UniverseScope {
  /** To be used when --infer_universe_scope applies. */
  public static final UniverseScope INFER_FROM_QUERY_EXPRESSION = new InferredUniverseScope();
  /** To be used when neither --universe_scope nor --infer_universe_scope are set. */
  public static final UniverseScope EMPTY = new ConstantUniverseScope(ImmutableList.of());

  private UniverseScope() {}

  /** To be used when --universe_scope is set. */
  public static UniverseScope fromUniverseScopeList(ImmutableList<String> constantUniverseScope) {
    return new ConstantUniverseScope(constantUniverseScope);
  }

  public abstract boolean isEmpty();

  public abstract Optional<ImmutableList<String>> getConstantValueMaybe();

  public abstract ImmutableList<String> inferFromQueryExpression(QueryExpression expr);

  private static final class ConstantUniverseScope extends UniverseScope {
    private final ImmutableList<String> constantUniverseScope;

    private ConstantUniverseScope(ImmutableList<String> constantUniverseScope) {
      this.constantUniverseScope = constantUniverseScope;
    }

    @Override
    public boolean isEmpty() {
      return constantUniverseScope.isEmpty();
    }

    @Override
    public Optional<ImmutableList<String>> getConstantValueMaybe() {
      return Optional.of(constantUniverseScope);
    }

    @Override
    public ImmutableList<String> inferFromQueryExpression(QueryExpression expr) {
      return constantUniverseScope;
    }
  }

  private static final class InferredUniverseScope extends UniverseScope {
    @Override
    public boolean isEmpty() {
      return false;
    }

    @Override
    public Optional<ImmutableList<String>> getConstantValueMaybe() {
      return Optional.empty();
    }

    @Override
    public ImmutableList<String> inferFromQueryExpression(QueryExpression expr) {
      LinkedHashSet<String> targetPatterns = new LinkedHashSet<>();
      expr.collectTargetPatterns(targetPatterns);
      return ImmutableList.copyOf(targetPatterns);
    }
  }
}
