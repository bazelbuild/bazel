// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;

/**
 * Marker interface for indicating a freezable Starlark value whose mutation operations check a
 * {@link Mutability}.
 */
// TODO(adonovan): merge into Freezable.
public interface StarlarkMutable extends Freezable, StarlarkValue {

  /**
   * Throws an exception if this object is not mutable.
   *
   * <p>This method is essentially a mix-in. Subclasses should not override it.
   *
   * @throws EvalException if the object is not mutable. This may be because the {@code
   *     this.mutability()} is frozen, or because it is temporarily locked by an active loop
   *     iteration.
   */
  default void checkMutable() throws EvalException {
    try {
      Mutability.checkMutable(this, mutability());
    } catch (MutabilityException ex) {
      throw new EvalException(null, ex.getMessage());
    }
  }
}
