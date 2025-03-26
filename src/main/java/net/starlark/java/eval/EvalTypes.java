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

package net.starlark.java.eval;

import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.Identifier;

/** Evaluates type annotations. */
final class EvalTypes {
  private EvalTypes() {} // uninstantiable

  static StarlarkType evalType(Module module, Expression expr) throws EvalException {
    switch (expr.kind()) {
      case IDENTIFIER:
        Identifier id = (Identifier) expr;
        Object result = Types.TYPE_UNIVERSE.get(id.getName());
        if (result == null) {
          throw Starlark.errorf("type '%s' is not defined", id.getName());
        }
        if (result instanceof StarlarkType type) {
          return type;
        }
      // TODO(ilist@): full evaluation: type expressions, applications
      // fall through
      default:
    }
    throw Starlark.errorf("expression '%s' is not a valid type.", expr);
  }
}
