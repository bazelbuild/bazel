// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import java.util.Objects;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;

/** Starlark function that determines whether aspect should be propagated to the target. */
public final class AspectPropagationPredicate {

  @SuppressWarnings("unused")
  private final StarlarkFunction predicate;

  @SuppressWarnings("unused")
  private final StarlarkSemantics semantics;

  public AspectPropagationPredicate(StarlarkFunction predicate, StarlarkSemantics semantics) {
    this.predicate = predicate;
    this.semantics = semantics;
  }

  public boolean evaluate() {
    // TODO(b/394400334): Add implementation depending on a given target and the starlark function.
    return true;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    AspectPropagationPredicate that = (AspectPropagationPredicate) o;
    return Objects.equals(predicate, that.predicate) && Objects.equals(semantics, that.semantics);
  }

  @Override
  public int hashCode() {
    return Objects.hash(predicate, semantics);
  }
}
