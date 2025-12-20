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

package net.starlark.java.syntax;

import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.FormatMethod;
import java.util.List;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;

/**
 * A visitor for validating that expressions and statements respect the types of the symbols
 * appearing within them, as determined by the type resolver.
 *
 * <p>Type annotations are not traversed by this visitor.
 */
public final class TypeChecker extends NodeVisitor {

  private final List<SyntaxError> errors;

  // Formats and reports an error at the start of the specified node.
  @FormatMethod
  private void errorf(Node node, String format, Object... args) {
    errorf(node.getStartLocation(), format, args);
  }

  // Formats and reports an error at the specified location.
  @FormatMethod
  private void errorf(Location loc, String format, Object... args) {
    errors.add(new SyntaxError(loc, String.format(format, args)));
  }

  private TypeChecker(List<SyntaxError> errors) {
    this.errors = errors;
  }

  /**
   * Returns the annotated type of an identifier's symbol, asserting that the binding information is
   * present.
   *
   * <p>If a type is not set on the binding it is taken to be {@code Any}.
   */
  // TODO: #27370 - An unannotated variable should either be treated as Any or else inferred from
  // its first binding occurrence, depending on how the var is introduced and whether it's in typed
  // code.
  private StarlarkType getType(Identifier id) {
    Resolver.Binding binding = id.getBinding();
    Preconditions.checkNotNull(binding);
    StarlarkType type = binding.getType();
    return type != null ? type : Types.ANY;
  }

  /**
   * Infers the type of an expression from a bottom-up traversal, relying on type information stored
   * in identifier bindings by the {@link TypeResolver}.
   *
   * <p>May not be called on type expressions (annotations, var statements, type alias statements).
   */
  private StarlarkType infer(Expression expr) {
    switch (expr.kind()) {
      case IDENTIFIER -> {
        var id = (Identifier) expr;
        return switch (id.getName()) {
          // As a hack, we special-case the names of these universal symbols that should really be
          // keywords.
          // TODO: #27728 - Instead of special casing, ensure type information is stored correctly
          // for the universal/predeclared symbols in the module, and retrieve it from there at type
          // resolution time. Then here we just need to get it from the binding like anything else.
          case "True", "False" -> Types.BOOL;
          case "None" -> Types.NONE;
          default -> getType(id);
        };
      }
      case STRING_LITERAL -> {
        return Types.STR;
      }
      case INT_LITERAL -> {
        return Types.INT;
      }
      case FLOAT_LITERAL -> {
        return Types.FLOAT;
      }
      default -> {
        // TODO: #28037 - support binaryop, call, cast, comprehension, conditional, dict_expr, dot,
        // index, lambda, list, and unaryop expressions.
        throw new UnsupportedOperationException(
            String.format("cannot typecheck %s expression", expr.kind()));
      }
    }
  }

  // Expressions should only be visited via infer(), not the visit() dispatch mechanism.
  // Override visit(Identifier) as a poison pill.
  @Override
  public void visit(Identifier id) {
    throw new AssertionError(
        String.format(
            "TypeChecker#visit should not have reached Identifier node '%s'", id.getName()));
  }

  @Override
  public void visit(AssignmentStatement assignment) {
    if (assignment.isAugmented()) {
      // TODO: #28037 - support this by validating that `lhs <op> rhs` would type check
      throw new UnsupportedOperationException("cannot typecheck augmented assignment statements");
    }
    if (!(assignment.getLHS() instanceof Identifier id)) {
      // TODO: #28037 - support LHSs containing multiple targets (list expression), field
      // assignments, and subscript assignments.
      throw new UnsupportedOperationException(
          "cannot typecheck assignment statements with anything besides a single identifier on the"
              + " LHS");
    }
    var lhsType = getType(id);
    // TODO: #27370 - Do bidirectional inference, passing down information about the expected type
    // from the LHS to the infer() call here, e.g. to construct the type of `[1, 2, 3]` as list[int]
    // instead of list[object].
    var rhsType = infer(assignment.getRHS());
    if (!StarlarkType.assignableFrom(lhsType, rhsType)) {
      errorf(assignment, "cannot assign type '%s' to '%s'", rhsType, lhsType);
    }
  }

  @Override
  public void visit(ExpressionStatement expr) {
    // Check constraints in the expression, but ignore the resulting type.
    // Don't dispatch to it via visit().
    infer(expr.getExpression());
  }

  // No need to override visit() for FlowStatement.

  @Override
  public void visit(LoadStatement load) {
    // Don't descend into children.
  }

  @Override
  public void visit(TypeAliasStatement alias) {
    // Don't descend into children.
  }

  @Override
  public void visit(VarStatement var) {
    // Don't descend into children.
  }

  // TODO: #28037 - Support `for`, `def`, `if`, and `return` statements.

  /**
   * Checks that the given file's AST satisfies the types in the bindings of its identifiers.
   *
   * <p>The file must have already been passed through the type resolver without error
   *
   * <p>Any type checking errors are appended to the file's errors list.
   */
  public static void check(StarlarkFile file) {
    TypeChecker checker = new TypeChecker(file.errors);
    checker.visit(file);
  }
}
