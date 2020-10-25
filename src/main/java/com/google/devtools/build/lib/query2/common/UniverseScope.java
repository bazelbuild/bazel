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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.skyframe.PrepareDepsOfPatternsValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.LinkedHashSet;
import java.util.Optional;
import javax.annotation.Nullable;

/** Representation of the --universe_scope option value. */
public interface UniverseScope {
  /** To be used when --infer_universe_scope applies. */
  UniverseScope INFER_FROM_QUERY_EXPRESSION = new InferredUniverseScope();
  /** To be used when neither --universe_scope nor --infer_universe_scope are set. */
  UniverseScope EMPTY = new ConstantUniverseScope(ImmutableList.of());

  /** To be used when --universe_scope is set. */
  static UniverseScope fromUniverseScopeList(ImmutableList<String> constantUniverseScope) {
    return new ConstantUniverseScope(constantUniverseScope);
  }

  UniverseSkyKey getUniverseKey(@Nullable QueryExpression expr, PathFragment offset);

  boolean isEmpty();

  Optional<ImmutableList<String>> getConstantValueMaybe();

  /** Constant universe scope. */
  final class ConstantUniverseScope implements UniverseScope {
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
    public UniverseSkyKey getUniverseKey(@Nullable QueryExpression expr, PathFragment offset) {
      return PrepareDepsOfPatternsValue.key(constantUniverseScope, offset);
    }
  }

  /** Universe scope inferred from query expression. */
  final class InferredUniverseScope implements UniverseScope {
    private InferredUniverseScope() {}

    @Override
    public boolean isEmpty() {
      return false;
    }

    @Override
    public Optional<ImmutableList<String>> getConstantValueMaybe() {
      return Optional.empty();
    }

    @Override
    public UniverseSkyKey getUniverseKey(@Nullable QueryExpression expr, PathFragment offset) {
      LinkedHashSet<String> targetPatterns = new LinkedHashSet<>();
      checkNotNull(expr).collectTargetPatterns(targetPatterns);
      return PrepareDepsOfPatternsValue.key(ImmutableList.copyOf(targetPatterns), offset);
    }
  }

}
