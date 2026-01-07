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
import java.util.ArrayList;
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

  private void errorIfKeyNotInt(IndexExpression index, StarlarkType objType, StarlarkType keyType) {
    if (!StarlarkType.assignableFrom(Types.INT, keyType)) {
      errorf(
          index.getLbracketLocation(),
          "'%s' of type '%s' must be indexed by an integer, but got '%s'",
          index.getObject(),
          objType,
          keyType);
    }
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
      case DOT -> {
        // TODO: #27370 - Add support for field retrieval on types besides Any.
        var dot = (DotExpression) expr;
        StarlarkType objType = infer(dot.getObject());
        if (objType.equals(Types.ANY)) {
          return Types.ANY;
        } else {
          errorf(
              dot.getDotLocation(),
              "'%s' of type '%s' does not have field '%s'",
              dot.getObject(),
              objType,
              dot.getField().getName());
          return Types.ANY;
        }
      }
      case INDEX -> {
        var index = (IndexExpression) expr;
        StarlarkType objType = infer(index.getObject());
        StarlarkType keyType = infer(index.getKey());

        // TODO: #28043 - Broaden list to Sequence and dict to Mapping, once we have better type
        // hierarchy support in the static type machinery.
        if (objType.equals(Types.ANY)) {
          return Types.ANY;
        } else if (objType instanceof Types.TupleType tupleType) {
          // TODO: #28037 - Support indexing tuples.
          throw new UnsupportedOperationException("cannot typecheck index expression on a tuple");
        } else if (objType instanceof Types.ListType listType) {
          errorIfKeyNotInt(index, objType, keyType); // fall through on error
          return listType.getElementType();
        } else if (objType instanceof Types.DictType dictType) {
          if (!StarlarkType.assignableFrom(dictType.getKeyType(), keyType)) {
            errorf(
                index.getLbracketLocation(),
                "'%s' of type '%s' requires key type '%s', but got '%s'",
                index.getObject(),
                objType,
                dictType.getKeyType(),
                keyType);
            // Fall through to returning the value type.
          }
          return dictType.getValueType();
        } else if (objType.equals(Types.STR)) {
          errorIfKeyNotInt(index, objType, keyType); // fall through on error
          return Types.STR;
        } else {
          errorf(
              index.getLbracketLocation(),
              "cannot index '%s' of type '%s'",
              index.getObject(),
              objType);
          return Types.ANY;
        }
      }
      default -> {
        // TODO: #28037 - support binaryop, call, cast, comprehension, conditional, dict_expr,
        // lambda, list, slice, and unaryop expressions.
        throw new UnsupportedOperationException(
            String.format("cannot typecheck %s expression", expr.kind()));
      }
    }
  }

  /**
   * Infers the type of an expression.
   *
   * <p>The expression must have already been resolved and type-resolved, i.e. type information must
   * be present in the identifiers' bindings.
   *
   * @throws SyntaxError.Exception if a static type error is present in the expression
   */
  static StarlarkType inferTypeOf(Expression expr) throws SyntaxError.Exception {
    List<SyntaxError> errors = new ArrayList<>();
    TypeChecker tc = new TypeChecker(errors);
    StarlarkType result = tc.infer(expr);
    if (!errors.isEmpty()) {
      throw new SyntaxError.Exception(tc.errors);
    }
    return result;
  }

  /**
   * Recursively typechecks the assignment of type {@code rhsType} to the target expression {@code
   * lhs}.
   *
   * <p>The asymmetry of the parameter types comes from the fact that this helper recursively
   * decomposes the LHS syntactically, whereas the RHS has already been fully evaluated to a type.
   * For instance, {@code x, y = (1, 2)} and {@code x, y = my_pair} both trigger the same behavior
   * in this method. Decomposing the LHS syntactically rather than by type is what allows {@code (x,
   * y) = [1, 2]} to succeed, even though assignment of a list to a tuple type is illegal (as in
   * {@code t : Tuple[int, int] = [1, 2]}).
   */
  private void assign(Expression lhs, StarlarkType rhsType) {
    // infer() handles Identifier and DotExpression. The type for evaluating these expressions in a
    // read context is the same as its type for assignment purposes.
    StarlarkType lhsType = infer(lhs);

    if (lhs.kind() == Expression.Kind.LIST_EXPR) {
      // TODO: #28037 - support LHSs containing multiple targets (list expression), field
      // assignments, and subscript assignments.
      throw new UnsupportedOperationException(
          "cannot typecheck assignment statements with multiple targets on the LHS");
    }

    if (!StarlarkType.assignableFrom(lhsType, rhsType)) {
      errorf(
          lhs.getStartLocation(),
          "cannot assign type '%s' to '%s' of type '%s'",
          rhsType,
          lhs,
          lhsType);
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

    // TODO: #27370 - Do bidirectional inference, passing down information about the expected type
    // from the LHS to the infer() call here, e.g. to construct the type of `[1, 2, 3]` as list[int]
    // instead of list[object].
    // TODO: #28037 - Consider rejecting the assignment if the LHS is read-only, e.g. an index
    // expression of a string. This would require either an ad hoc check here in this method, or
    // else passing back from infer() more detailed information than just the StarlarkType.
    var rhsType = infer(assignment.getRHS());

    assign(assignment.getLHS(), rhsType);
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
