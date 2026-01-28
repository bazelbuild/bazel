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
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A visitor for validating that expressions and statements respect the types of the symbols
 * appearing within them, as determined by the type tagger.
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

  private void binaryOperatorError(
      BinaryOperatorExpression binop, StarlarkType xType, StarlarkType yType) {
    // TODO: #28037 - better error message if LHS and/or RHS are unions?
    errorf(
        binop.getOperatorLocation(),
        "operator '%s' cannot be applied to types '%s' and '%s'",
        binop.getOperator(),
        xType,
        yType);
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
   * in identifier bindings by the {@link TypeTagger}.
   *
   * <p>May not be called on type expressions (annotations, var statements, type alias statements).
   */
  private StarlarkType infer(Expression expr) {
    switch (expr.kind()) {
      case IDENTIFIER -> {
        return getType((Identifier) expr);
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
        var dot = (DotExpression) expr;
        StarlarkType objType = infer(dot.getObject());
        String name = dot.getField().getName();
        StarlarkType fieldType = objType.getField(name);
        if (fieldType == null) {
          errorf(
              dot.getDotLocation(),
              "'%s' of type '%s' does not have field '%s'",
              dot.getObject(),
              objType,
              name);
          return Types.ANY;
        }
        return fieldType;
      }
      case INDEX -> {
        return inferIndex((IndexExpression) expr);
      }
      case LIST_EXPR -> {
        var list = (ListExpression) expr;
        List<StarlarkType> elementTypes = new ArrayList<>();
        for (Expression element : list.getElements()) {
          elementTypes.add(infer(element));
        }
        return list.isTuple()
            ? Types.tuple(ImmutableList.copyOf(elementTypes))
            : Types.list(Types.union(elementTypes));
      }
      case DICT_EXPR -> {
        var dict = (DictExpression) expr;
        List<StarlarkType> keyTypes = new ArrayList<>();
        List<StarlarkType> valueTypes = new ArrayList<>();
        for (var entry : dict.getEntries()) {
          keyTypes.add(infer(entry.getKey()));
          valueTypes.add(infer(entry.getValue()));
        }
        return Types.dict(Types.union(keyTypes), Types.union(valueTypes));
      }
      case BINARY_OPERATOR -> {
        var binop = (BinaryOperatorExpression) expr;
        TokenKind operator = binop.getOperator();
        switch (operator) {
          case AND, OR, EQUALS_EQUALS, NOT_EQUALS -> {
            // Boolean regardless of LHS and RHS.
            return Types.BOOL;
          }
          case LESS, LESS_EQUALS, GREATER, GREATER_EQUALS -> {
            // Boolean or type error.
            StarlarkType xType = infer(binop.getX());
            StarlarkType yType = infer(binop.getY());
            if (StarlarkType.comparable(xType, yType)) {
              return Types.BOOL;
            }
            binaryOperatorError(binop, xType, yType);
            return Types.ANY;
          }
          default -> {
            // Take the union of all types inferred by crossing the left and right union elements
            // (each of which must be a valid combination of rhs and lhs for the operator).
            StarlarkType xType = infer(binop.getX());
            StarlarkType yType = infer(binop.getY());
            ImmutableCollection<StarlarkType> xTypes = Types.unfoldUnion(xType);
            ImmutableCollection<StarlarkType> yTypes = Types.unfoldUnion(yType);
            ArrayList<StarlarkType> resultTypes = new ArrayList<>();
            for (StarlarkType xElemType : xTypes) {
              for (StarlarkType yElemType : yTypes) {
                @Nullable
                StarlarkType resultType = xElemType.inferBinaryOperator(operator, yElemType, true);
                if (resultType == null) {
                  resultType = yElemType.inferBinaryOperator(operator, xElemType, false);
                }
                if (resultType == null && operator == TokenKind.STAR) {
                  // Tuple repetition is the only case where we need to examine the expressions.
                  // TODO: #28037 - We can get rid of the tuple repetition special case if we
                  // introduce ConstantIntType for integer constants.
                  if (xElemType.equals(Types.INT) && yElemType instanceof Types.TupleType tuple) {
                    resultType = inferTupleRepetition(tuple, binop.getX());
                  } else if (yElemType.equals(Types.INT)
                      && xElemType instanceof Types.TupleType tuple) {
                    resultType = inferTupleRepetition(tuple, binop.getY());
                  }
                }
                if (resultType == null) {
                  binaryOperatorError(binop, xType, yType);
                  return Types.ANY;
                }
                resultTypes.add(resultType);
              }
            }
            return Types.union(resultTypes);
          }
        }
      }
      case UNARY_OPERATOR -> {
        var unop = (UnaryOperatorExpression) expr;
        if (unop.getOperator() == TokenKind.NOT) {
          // NOT always returns a boolean (even if applied to Any or unions).
          return Types.BOOL;
        }
        StarlarkType xType = infer(unop.getX());
        if (xType.equals(Types.ANY)
            || ((unop.getOperator() == TokenKind.MINUS || unop.getOperator() == TokenKind.PLUS)
                && isNumeric(xType))
            || (unop.getOperator() == TokenKind.TILDE && xType.equals(Types.INT))) {
          // Unary operators other than NOT preserve the type of their operand.
          return xType;
        }
        errorf(
            unop.getStartLocation(),
            "operator '%s' cannot be applied to type '%s'",
            unop.getOperator(),
            xType);
        return Types.ANY;
      }
      default -> {
        // TODO: #28037 - support call, cast, comprehension, conditional, lambda, and slice
        // expressions.
        errorf(expr, "UNSUPPORTED: cannot typecheck %s expression", expr.kind());
        return Types.ANY;
      }
    }
  }

  private static boolean isNumeric(StarlarkType type) {
    if (type.equals(Types.INT) || type.equals(Types.FLOAT)) {
      return true;
    }
    if (type instanceof Types.UnionType unionType) {
      return unionType.getTypes().stream().allMatch(TypeChecker::isNumeric);
    }
    return false;
  }

  private StarlarkType inferIndex(IndexExpression index) {
    Expression obj = index.getObject();
    Expression key = index.getKey();
    StarlarkType objType = infer(obj);
    StarlarkType keyType = infer(key);

    if (objType.equals(Types.ANY)) {
      return Types.ANY;

    } else if (objType instanceof Types.TupleType tupleType) {
      errorIfKeyNotInt(index, objType, keyType);
      var elementTypes = tupleType.getElementTypes();
      StarlarkType resultType = null;
      // Project out the type of the specific component if we can statically determine the index.
      // TODO: #28037 - Consider allowing more complicated static expressions, e.g. unary
      // (minus sign) and binary operators on integers.
      if (key.kind() == Expression.Kind.INT_LITERAL) {
        Integer i = ((IntLiteral) key).getIntValueExact();
        if (i != null) {
          if (0 <= i && i < elementTypes.size()) {
            resultType = elementTypes.get(i);
          } else {
            errorf(
                index.getLbracketLocation(),
                "'%s' of type '%s' is indexed by integer %s, which is out-of-range",
                obj,
                objType,
                i);
            // Don't complain about uses of the result type when we don't even know what result type
            // the user wanted.
            return Types.ANY;
          }
        }
      }
      if (resultType == null) {
        resultType = Types.union(elementTypes);
      }
      return resultType;

      // TODO: #28043 - Broaden from List to Sequence once we have better type hierarchy support.
    } else if (objType instanceof Types.ListType listType) {
      errorIfKeyNotInt(index, objType, keyType); // fall through on error
      return listType.getElementType();

      // TODO: #28043 - Broaden from Dict to Mapping once we have better type hierarchy support.
    } else if (objType instanceof Types.DictType dictType) {
      if (!StarlarkType.assignableFrom(dictType.getKeyType(), keyType)) {
        errorf(
            index.getLbracketLocation(),
            "'%s' of type '%s' requires key type '%s', but got '%s'",
            obj,
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
      errorf(index.getLbracketLocation(), "cannot index '%s' of type '%s'", obj, objType);
      return Types.ANY;
    }
  }

  private static StarlarkType inferTupleRepetition(Types.TupleType tuple, Expression timesExpr) {
    if (timesExpr instanceof IntLiteral intLiteral) {
      // TODO: #28037 - our IntLiteral is always non-negative (we parse negative integers as unary
      // expressions). Note, however, that mypy does handle negative integers; we ought to either
      // support negative integers in ConstantIntType if/when we introduce it, or else optimize
      // negative integer literals to be IntLiteral (see #28385).
      @Nullable Integer times = intLiteral.getIntValueExact();
      if (times != null) {
        return tuple.repeat(times);
      }
    }
    // TODO: #28037 - return tuple of indeterminate shape.
    return Types.ANY;
  }

  /**
   * Infers the type of an expression.
   *
   * <p>The expression must have already been resolved and type-tagged, i.e. identifiers must have
   * their bindings set and these bindings must contain type information.
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
      errorf(
          lhs,
          "UNSUPPORTED: cannot typecheck assignment statements with multiple targets on the LHS");
      return;
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
      errorf(assignment, "UNSUPPORTED: cannot typecheck augmented assignment statements");
      return;
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
   * <p>The file must have already been passed through the {@link TypeTagger} without error
   *
   * <p>Any type checking errors are appended to the file's errors list.
   */
  public static void checkFile(StarlarkFile file) {
    TypeChecker checker = new TypeChecker(file.errors);
    checker.visit(file);
  }
}
