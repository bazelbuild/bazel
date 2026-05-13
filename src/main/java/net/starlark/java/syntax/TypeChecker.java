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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.spelling.SpellChecker;

/**
 * A visitor for validating that expressions and statements respect the types of the symbols
 * appearing within them, as determined by the type tagger.
 *
 * <p>In addition, this visitor modifies the function type on the {@link Resolver.Function} objects
 * of {@link LambdaExpression}s in the {@link TypeTable} (originally populated by the {@link
 * TypeTagger}) to have a more precise return type, if possible; and populates the types of the
 * {@link Resolver.Binding} objects of untyped variables with the inferred types of their values in
 * their first assignments in typed code.
 *
 * <p>Type annotations are not traversed by this visitor.
 */
public final class TypeChecker extends NodeVisitor {

  private final TypeTable typeTable;
  private final TypeContext typeContext;

  // Empty if we were invoked via inferTypeOf() to type-check an expression (since inside
  // an expression, no function definitions are allowed). Populated and mutated by visitation.
  private final ArrayDeque<Resolver.Function> functionStack = new ArrayDeque<>();

  // Formats and reports an error at the start of the specified node.
  @FormatMethod
  private void errorf(Node node, String format, Object... args) {
    errorf(node.getStartLocation(), format, args);
  }

  // Formats and reports an error at the specified location.
  @FormatMethod
  private void errorf(Location loc, String format, Object... args) {
    typeTable.errors.add(new SyntaxError(loc, String.format(format, args)));
  }

  private void binaryOperatorError(
      StarlarkType xType,
      TokenKind operator,
      Location operatorLocation,
      StarlarkType yType,
      boolean augmentedAssignment,
      String extraMessage) {
    // TODO: #28037 - better error message if LHS and/or RHS are unions?
    errorf(
        operatorLocation,
        "operator '%s%s' cannot be applied to types '%s' and '%s'%s",
        operator,
        augmentedAssignment ? "=" : "",
        xType,
        yType,
        extraMessage.isEmpty() ? "" : ": " + extraMessage);
  }

  private void binaryOperatorError(
      StarlarkType xType,
      TokenKind operator,
      Location operatorLocation,
      StarlarkType yType,
      boolean augmentedAssignment) {
    binaryOperatorError(xType, operator, operatorLocation, yType, augmentedAssignment, "");
  }

  private static String plural(int n) {
    return n == 1 ? "" : "s";
  }

  private TypeChecker(TypeTable typeTable, TypeContext typeContext) {
    this.typeTable = typeTable;
    this.typeContext = typeContext;
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
    checkNotNull(binding);
    StarlarkType type = typeTable.getType(binding);
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
      case CAST -> {
        var cast = (CastExpression) expr;
        var unused = infer(cast.getValue()); // only to verify the value expr is well-typed
        return cast.getStarlarkType();
      }
      case DOT -> {
        return inferDot((DotExpression) expr);
      }
      case INDEX -> {
        return inferIndex((IndexExpression) expr);
      }
      case SLICE -> {
        return inferSlice((SliceExpression) expr);
      }
      case LAMBDA -> {
        var lambda = (LambdaExpression) expr;
        StarlarkType inferedReturnType = infer(lambda.getBody());
        Types.CallableType originalType =
            checkNotNull(
                typeTable.getType(lambda.getResolvedFunction()),
                "type tagger should have set type for lambda expr '%s'",
                lambda);
        if (!originalType.getReturnType().equals(inferedReturnType)) {
          // Update the lambda function type with a more precise return type.
          typeTable.setType(
              lambda.getResolvedFunction(),
              Types.callable(
                  originalType.getParameterNames(),
                  originalType.getParameterTypes(),
                  originalType.getNumPositionalOnlyParameters(),
                  originalType.getNumPositionalParameters(),
                  originalType.getMandatoryParameters(),
                  originalType.getVarargsType(),
                  originalType.getKwargsType(),
                  inferedReturnType));
        }
        return typeTable.getType(lambda.getResolvedFunction());
      }
      case LIST_EXPR -> {
        var list = (ListExpression) expr;
        List<StarlarkType> elementTypes = new ArrayList<>();
        for (Expression element : list.getElements()) {
          elementTypes.add(infer(element));
        }
        return list.isTuple()
            ? Types.tuple(ImmutableList.copyOf(elementTypes))
            : Types.listRvalue(Types.union(elementTypes));
      }
      case DICT_EXPR -> {
        var dict = (DictExpression) expr;
        List<StarlarkType> keyTypes = new ArrayList<>();
        List<StarlarkType> valueTypes = new ArrayList<>();
        for (var entry : dict.getEntries()) {
          keyTypes.add(infer(entry.getKey()));
          valueTypes.add(infer(entry.getValue()));
        }
        return Types.dictRvalue(Types.union(keyTypes), Types.union(valueTypes));
      }
      case CALL -> {
        // TODO: #27370 - we could special-case set literals; e.g. check if a call expression is
        // `set()`, verifying using typeContext that `set` is the set type constructor.
        return inferCall((CallExpression) expr);
      }
      case CONDITIONAL -> {
        var cond = (ConditionalExpression) expr;
        return Types.union(infer(cond.getThenCase()), infer(cond.getElseCase()));
      }
      case BINARY_OPERATOR -> {
        var binop = (BinaryOperatorExpression) expr;
        StarlarkType xType = infer(binop.getX());
        StarlarkType yType = infer(binop.getY());
        return inferBinaryOperator(
            binop.getX(),
            xType,
            binop.getOperator(),
            binop.getOperatorLocation(),
            binop.getY(),
            yType,
            /* augmentedAssignment= */ false);
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
                && StarlarkType.assignableFrom(Types.NUMERIC, xType))
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
      case COMPREHENSION -> {
        return inferComprehension((Comprehension) expr);
      }
      default -> {
        // TODO: #28037 - support isinstance expressions.
        errorf(expr, "UNSUPPORTED: cannot typecheck %s expression", expr.kind());
        return Types.ANY;
      }
    }
  }

  /**
   * Returns the integer value of an expression if it's an integer value which can be exactly
   * represented as a Java integer, or null otherwise (in particular, if the expression itself is
   * null).
   */
  @Nullable
  private static Integer getIntValueExact(@Nullable Expression expr) {
    if (expr instanceof IntLiteral intLiteral) {
      return intLiteral.getIntValueExact();
    }
    return null;
  }

  private StarlarkType inferDot(DotExpression dot) {
    return Types.union(inferDotUnfolded(dot, infer(dot.getObject())));
  }

  /**
   * Infers the non-flattened unfolded list of possible types of a dot expression.
   *
   * <p>For example, given if field f has type int for type T, and type str|bool for type U, this
   * function will return the list {@code [int, str|bool]} for x.f where x has type T|U.
   *
   * <p>When a dot expression is used as a value, one should take the union type of the returned
   * types. But when a dot expression is used as the LHS of an assignment, one should take their
   * meet.
   */
  private ImmutableList<StarlarkType> inferDotUnfolded(DotExpression dot, StarlarkType objType) {
    String name = dot.getField().getName();

    if (objType.equals(Types.ANY)) {
      return ImmutableList.of(Types.ANY);
    }

    ImmutableCollection<StarlarkType> objElemTypes = Types.unfoldUnion(objType);
    ImmutableList.Builder<StarlarkType> resultTypes =
        ImmutableList.builderWithExpectedSize(objElemTypes.size());
    for (StarlarkType objElemType : objElemTypes) {
      StarlarkType fieldType = objElemType.getField(name, typeContext);
      if (fieldType == null) {
        errorf(
            dot.getDotLocation(),
            "'%s' of type '%s' does not have field '%s'",
            dot.getObject(),
            objType,
            name);
        return ImmutableList.of(Types.ANY);
      }
      resultTypes.add(fieldType);
    }
    return resultTypes.build();
  }

  private StarlarkType inferIndex(IndexExpression index) {
    return Types.union(inferIndexUnfolded(index, infer(index.getObject()), infer(index.getKey())));
  }

  /**
   * Infers the non-flattened unfolded list of possible types of an index expression.
   *
   * <p>For example, given object type {@code list[int] | list[str|bool]}, this function will return
   * the list {@code [int, str|bool]}.
   *
   * <p>When an index expression is used as a value, one should take the union type of the returned
   * types. But when an index expression is used as the LHS of an assignment, one should take their
   * meet.
   */
  private ImmutableList<StarlarkType> inferIndexUnfolded(
      IndexExpression index, StarlarkType objType, StarlarkType keyType) {
    Expression obj = index.getObject();
    Expression key = index.getKey();

    if (objType.equals(Types.ANY)) {
      return ImmutableList.of(Types.ANY);
    }

    ImmutableCollection<StarlarkType> objElemTypes = Types.unfoldUnion(objType);
    ImmutableList.Builder<StarlarkType> resultTypes =
        ImmutableList.builderWithExpectedSize(objElemTypes.size());
    for (StarlarkType objElemType : objElemTypes) {
      if (objElemType.equals(Types.ANY)) {
        resultTypes.add(Types.ANY);

      } else if (objElemType instanceof Types.FixedLengthTupleType tupleType) {
        errorIfKeyNotInt(index, objElemType, keyType);
        var elementTypes = tupleType.getElementTypes();
        StarlarkType resultType = null;
        // Project out the type of the specific component if we can statically determine the index.
        Integer intKey = getIntValueExact(key);
        if (intKey != null) {
          int i = intKey;
          if (i < 0) {
            // Same logic as for EvalUtils#getSequenceIndex.
            i += elementTypes.size();
          }
          if (0 <= i && i < elementTypes.size()) {
            resultType = elementTypes.get(i);
          } else {
            errorf(
                index.getLbracketLocation(),
                "'%s' of type '%s' is indexed by integer %s, which is out-of-range",
                obj,
                objType,
                intKey);
            // Don't complain about uses of the result type when we don't even know what result type
            // the user wanted.
            return ImmutableList.of(Types.ANY);
          }
        }
        if (resultType == null) {
          resultType = tupleType.toHomogeneous().getElementType();
        }
        resultTypes.add(resultType);

      } else if (objElemType instanceof Types.AbstractSequenceType sequenceType) {
        errorIfKeyNotInt(index, objType, keyType); // fall through on error
        resultTypes.add(sequenceType.getElementType());

      } else if (objElemType instanceof Types.AbstractMappingType mappingType) {
        if (!StarlarkType.assignableFrom(mappingType.getKeyType(), keyType)) {
          errorf(
              index.getLbracketLocation(),
              "'%s' of type '%s' requires key type '%s', but got '%s'",
              obj,
              objType,
              mappingType.getKeyType(),
              keyType);
          // Fall through to returning the value type.
        }
        resultTypes.add(mappingType.getValueType());

      } else if (objElemType.equals(Types.STR)) {
        errorIfKeyNotInt(index, objType, keyType); // fall through on error
        resultTypes.add(Types.STR);

      } else {
        errorf(index.getLbracketLocation(), "cannot index '%s' of type '%s'", obj, objType);
        return ImmutableList.of(Types.ANY);
      }
    }
    return resultTypes.build();
  }

  private StarlarkType inferSlice(SliceExpression slice) {
    @Nullable Integer step = getIntValueExact(slice.getStep());
    if (step == null) {
      step = 1;
      if (slice.getStep() != null) {
        StarlarkType stepType = infer(slice.getStep());
        if (!StarlarkType.assignableFrom(Types.INT, stepType)) {
          errorf(slice.getStep(), "got '%s' for slice step, want int", stepType);
          return Types.ANY;
        }
      }
    } else if (step == 0) {
      errorf(slice.getStep(), "slice step cannot be zero");
      return Types.ANY;
    }
    if (slice.getStart() != null) {
      StarlarkType startType = infer(slice.getStart());
      if (!StarlarkType.assignableFrom(Types.INT, startType)) {
        errorf(slice.getStart(), "got '%s' for start index, want int", startType);
        return Types.ANY;
      }
    }
    if (slice.getStop() != null) {
      StarlarkType stopType = infer(slice.getStop());
      if (!StarlarkType.assignableFrom(Types.INT, stopType)) {
        errorf(slice.getStop(), "got '%s' for stop index, want int", stopType);
        return Types.ANY;
      }
    }

    StarlarkType objType = infer(slice.getObject());
    if (objType.equals(Types.ANY)) {
      return Types.ANY;
    }
    ArrayList<StarlarkType> resultTypes = new ArrayList<>();
    for (StarlarkType objElemType : Types.unfoldUnion(objType)) {
      if (objElemType.equals(Types.ANY)) {
        resultTypes.add(Types.ANY);
      } else if (objElemType.equals(Types.STR)) {
        resultTypes.add(Types.STR);
      } else if (objElemType instanceof Types.FixedLengthTupleType tupleType) {
        ImmutableList<StarlarkType> tupleElementTypes = tupleType.getElementTypes();
        int len = tupleElementTypes.size();
        @Nullable Integer start = getIntValueExact(slice.getStart());
        @Nullable Integer stop = getIntValueExact(slice.getStop());
        ImmutableList.Builder<StarlarkType> resultTupleElementTypes = ImmutableList.builder();
        if (step != null
            && haveExactSliceBound(slice.getStart(), start)
            && haveExactSliceBound(slice.getStop(), stop)) {
          if (step > 0) {
            int startClamped = start != null ? SyntaxUtils.toSliceBound(start, len) : 0;
            int stopClamped = stop != null ? SyntaxUtils.toSliceBound(stop, len) : len;
            for (long i = startClamped; i < stopClamped && (int) i == i; i += step) {
              resultTupleElementTypes.add(tupleElementTypes.get((int) i));
            }
          } else {
            int startClamped =
                start != null ? SyntaxUtils.toReverseSliceBound(start, len) : len - 1;
            int stopClamped = stop != null ? SyntaxUtils.toReverseSliceBound(stop, len) : -1;
            for (long i = startClamped; i > stopClamped && (int) i == i; i += step) {
              resultTupleElementTypes.add(tupleElementTypes.get((int) i));
            }
          }
          resultTypes.add(Types.tuple(resultTupleElementTypes.build()));
        } else {
          resultTypes.add(tupleType.toHomogeneous());
        }
      } else if (objElemType instanceof Types.AbstractSequenceType sequenceType) {
        resultTypes.add(sequenceType);
      } else {
        errorf(
            slice.getLbracketLocation(),
            "invalid slice operand '%s' of type '%s', expected Sequence or str",
            slice.getObject(),
            objElemType);
        resultTypes.add(Types.ANY);
      }
    }
    return Types.union(resultTypes);
  }

  private static boolean haveExactSliceBound(
      @Nullable Expression expr, @Nullable Integer exprIntValueExact) {
    if (expr == null) {
      // Bound not specified, so we know its exact value (the default value)
      return true;
    }
    if (exprIntValueExact != null) {
      // Bound is specified and is a 32-bit integer literal (or negation)
      return true;
    }
    return false;
  }

  private StarlarkType inferBinaryOperator(
      Expression xExpr,
      StarlarkType xType,
      TokenKind operator,
      Location operatorLocation,
      Expression yExpr,
      StarlarkType yType,
      boolean augmentedAssignment) {
    // TokenKind operator = binop.getOperator();
    switch (operator) {
      case AND, OR, EQUALS_EQUALS, NOT_EQUALS -> {
        // Boolean regardless of LHS and RHS.
        return Types.BOOL;
      }
      case LESS, LESS_EQUALS, GREATER, GREATER_EQUALS -> {
        // Boolean or type error.
        if (StarlarkType.comparable(xType, yType)) {
          return Types.BOOL;
        }
        binaryOperatorError(xType, operator, operatorLocation, yType, augmentedAssignment);
        return Types.ANY;
      }
      default -> {
        // Take the union of all types inferred by crossing the left and right union elements
        // (each of which must be a valid combination of rhs and lhs for the operator).
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
              if (StarlarkType.assignableFrom(Types.INT, xElemType)
                  && yElemType instanceof Types.TupleType tuple) {
                resultType = inferTupleRepetition(tuple, xExpr);
              } else if (StarlarkType.assignableFrom(Types.INT, yElemType)
                  && xElemType instanceof Types.TupleType tuple) {
                resultType = inferTupleRepetition(tuple, yExpr);
              }
            }
            if (resultType == null) {
              binaryOperatorError(xType, operator, operatorLocation, yType, augmentedAssignment);
              return Types.ANY;
            }
            resultTypes.add(resultType);
          }
        }
        return Types.union(resultTypes);
      }
    }
  }

  private StarlarkType inferCall(CallExpression call) {
    // Collect and check the shape of the call's *args/**kwargs. (This check is independent of
    // callFunctionType.)
    @Nullable VarargsArgument varargs = null;
    @Nullable KwargsArgument kwargs = null;
    int numArgs = call.getArguments().size();
    if (numArgs > 0 && call.getArguments().get(numArgs - 1) instanceof Argument.StarStar arg) {
      kwargs = KwargsArgument.of(arg, this);
      if (kwargs == null) {
        // error already reported
        return Types.ANY;
      }
      numArgs--;
    }
    if (numArgs > 0 && call.getArguments().get(numArgs - 1) instanceof Argument.Star arg) {
      varargs = VarargsArgument.of(arg, this);
      if (varargs == null) {
        // error already reported
        return Types.ANY;
      }
      numArgs--;
    }

    StarlarkType callFunctionType = infer(call.getFunction());
    if (callFunctionType.equals(Types.ANY)) {
      return Types.ANY;
    }

    // Collect call's argument types (excluding *args and **kwargs).
    ImmutableList<StarlarkType> argTypes =
        call.getArguments().stream()
            .limit(numArgs)
            .map(arg -> infer(arg.getValue()))
            .collect(toImmutableList());

    ImmutableCollection<StarlarkType> callFunctionTypes = Types.unfoldUnion(callFunctionType);
    ArrayList<StarlarkType> returnTypes = new ArrayList<>();
    for (StarlarkType callFunctionElemType : callFunctionTypes) {
      if (callFunctionElemType.equals(Types.ANY)) {
        // Nothing we can check.
        returnTypes.add(Types.ANY);
        continue;
      }
      Types.CallableType callable = callFunctionElemType instanceof Types.CallableType c ? c : null;
      if (callable == null) {
        errorf(
            call.getFunction(),
            "'%s' is not callable; got type '%s'",
            call.getFunction(),
            callFunctionType);
        return Types.ANY;
      }

      // TODO: #28043 - Some of the checks below can be used to implement
      // Types.CallableType.assignableFromHook().

      // Indices of residual arguments in call.getArguments() and their corresponding types in
      // argTypes. (Micro-optimization to avoid allocating <Argument, StarlarkType> pairs.)
      ArrayList<Integer> residualPositional = new ArrayList<>(0);
      ArrayList<Integer> residualNamed = new ArrayList<>(0);
      // Names of mandatory parameters (both positional and named) having a corresponding argument.
      ArrayList<String> seenMandatoryParameters =
          new ArrayList<>(callable.getMandatoryParameters().size());
      for (int i = 0; i < numArgs; i++) {
        Argument arg = call.getArguments().get(i);
        int parameterIndex;
        if (i < call.getNumPositionalArguments()) {
          // positional argument
          if (i < callable.getNumPositionalParameters()) {
            parameterIndex = i;
          } else {
            residualPositional.add(i);
            continue;
          }
        } else {
          // keyword argument
          parameterIndex = callable.getParameterNames().indexOf(arg.getName());
          if (parameterIndex < callable.getNumPositionalOnlyParameters()) {
            // Either no param was found (i<0) or it's positional-only (0<=i<numPosOnly).
            residualNamed.add(i);
            continue;
          }
        }
        // Argument is not residual; check it against the corresponding parameter.
        String parameterName = callable.getParameterNames().get(parameterIndex);
        StarlarkType parameterType = callable.getParameterTypeByPos(parameterIndex);
        if (callable.getMandatoryParameters().contains(parameterName)) {
          seenMandatoryParameters.add(parameterName);
        }
        if (!StarlarkType.assignableFrom(parameterType, argTypes.get(i))) {
          errorf(
              call.getArguments().get(i),
              "in call to '%s()', parameter '%s' got value of type '%s', want '%s'",
              call.getFunction(),
              parameterName,
              argTypes.get(i),
              parameterType);
          return Types.ANY;
        }
      }
      if (!checkCallResidualPositionals(residualPositional, call, callable, argTypes)
          || !checkCallResidualNamed(residualNamed, call, callable, argTypes)) {
        return Types.ANY;
      }
      if (!checkCallMissingMandatoryArgs(
          seenMandatoryParameters,
          /* callHasVarargs= */ varargs != null,
          /* callHasKwargs= */ kwargs != null,
          call,
          callable)) {
        return Types.ANY;
      }
      // Like mypy, we check that the call's *args/**kwargs values are assignable to the callable's
      // varargs/kwargs type. This is useful for the common case of a wrapper around a function
      // which forwards its *args/**kwargs to the wrapped function unchanged; but it also raises
      // failures for some legitimate uses: `def f(x: Any, **kwargs: str): ... ; f(**{"x" : 42})`.
      // In that case, the caller can bypass the check by casting to Any: `f(**(cast(Any, ...)))`.
      // We skip the check if the callable doesn't accept *args/**kwargs because the call's
      // *args/**kwargs may be used to set any remaining unset arguments, or may be empty.
      if (varargs != null
          && !checkAssignable(
              callable.getVarargsType(),
              varargs.elementType(),
              call,
              varargs.expr(),
              "elements of argument after *")) {
        return Types.ANY;
      }
      if (kwargs != null
          && !checkAssignable(
              callable.getKwargsType(),
              kwargs.valueType(),
              call,
              kwargs.expr(),
              "values of argument after **")) {
        return Types.ANY;
      }
      returnTypes.add(callable.getReturnType());
    }
    return Types.union(returnTypes);
  }

  private static record VarargsArgument(Expression expr, StarlarkType elementType) {
    @Nullable
    static VarargsArgument of(Argument.Star arg, TypeChecker checker) {
      Expression varargs = arg.getValue();
      StarlarkType varargsType = checker.infer(varargs);
      StarlarkType varargsElementType = findElementType(varargsType);
      if (varargsElementType == null) {
        checker.errorf(varargs, "argument after * must be a sequence, not '%s'", varargsType);
        return null;
      }
      return new VarargsArgument(varargs, varargsElementType);
    }

    /**
     * Finds the smallest {@code Sequence[E]} type which is a supertype of the given type, and
     * return E; or null if the given type does not have such a supertype.
     */
    @Nullable
    private static StarlarkType findElementType(StarlarkType maybeSequence) {
      if (maybeSequence.equals(Types.ANY)) {
        return Types.ANY;
      }
      ImmutableCollection<StarlarkType> unfolded = Types.unfoldUnion(maybeSequence);
      ArrayList<StarlarkType> elements = new ArrayList<>(unfolded.size());
      for (StarlarkType unfoldedElem : unfolded) {
        // TODO: #28037 - Check getSubtypes() instead of relying purely on Java inheritance.
        if (unfoldedElem instanceof Types.AbstractSequenceType sequence) {
          elements.add(sequence.getElementType());
        } else {
          return null;
        }
      }
      return Types.union(elements);
    }
  }

  private static record KwargsArgument(Expression expr, StarlarkType valueType) {
    @Nullable
    static KwargsArgument of(Argument.StarStar arg, TypeChecker checker) {
      Expression kwargs = arg.getValue();
      StarlarkType kwargsType = checker.infer(kwargs);
      StarlarkType kwargsValueType = findValueType(kwargsType);
      if (kwargsValueType == null) {
        checker.errorf(
            kwargs, "argument after ** must be a dict with string keys, not '%s'", kwargsType);
        return null;
      }
      return new KwargsArgument(kwargs, kwargsValueType);
    }

    /**
     * Finds the smallest {@code Mapping[K, V]} type which is a supertype of the given type such
     * that K is (a consistent-subtype-of?) str, and returns V; or null if the given type does not
     * have such a supertype.
     */
    @Nullable
    private static StarlarkType findValueType(StarlarkType maybeMapping) {
      if (maybeMapping.equals(Types.ANY)) {
        return Types.ANY;
      }
      ImmutableCollection<StarlarkType> unfolded = Types.unfoldUnion(maybeMapping);
      ArrayList<StarlarkType> values = new ArrayList<>(unfolded.size());
      for (StarlarkType unfoldedElem : unfolded) {
        // TODO: #28037 - Check getSubtypes() instead of relying purely on Java inheritance.
        if (unfoldedElem instanceof Types.AbstractMappingType mapping
            && StarlarkType.assignableFrom(Types.STR, mapping.getKeyType())) {
          values.add(mapping.getValueType());
        } else {
          return null;
        }
      }
      return Types.union(values);
    }
  }

  /**
   * Returns true if the call's residual positional arguments (if any) satisfy the type checker.
   * Otherwise, reports an error and returns false.
   */
  private boolean checkCallResidualPositionals(
      List<Integer> residualPositional,
      CallExpression call,
      Types.CallableType callable,
      List<StarlarkType> argTypes) {
    if (residualPositional.isEmpty()) {
      return true;
    } else if (callable.getVarargsType() == null) {
      // callable cannot accept residual positional args
      if (callable.getNumPositionalParameters() > 0) {
        errorf(
            call.getArguments().get(callable.getNumPositionalParameters()),
            "'%s()' accepts no more than %d positional argument%s but got %d",
            call.getFunction(),
            callable.getNumPositionalParameters(),
            plural(callable.getNumPositionalParameters()),
            call.getNumPositionalArguments());
      } else {
        errorf(
            call.getArguments().getFirst(),
            "'%s()' does not accept positional arguments, but got %d",
            call.getFunction(),
            call.getNumPositionalArguments());
      }
      return false;
    } else {
      // residual positional args go into callable's varargs
      for (int argIndex : residualPositional) {
        Argument arg = call.getArguments().get(argIndex);
        StarlarkType argType = argTypes.get(argIndex);
        if (!checkAssignable(
            callable.getVarargsType(), argType, call, arg, "residual positional arguments")) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Returns true if the call's residual named arguments (if any) satisfy the type checker.
   * Otherwise, reports an error and returns false.
   */
  private boolean checkCallResidualNamed(
      List<Integer> residualNamed,
      CallExpression call,
      Types.CallableType callable,
      List<StarlarkType> argTypes) {
    if (residualNamed.isEmpty()) {
      return true;
    } else if (callable.getKwargsType() == null) {
      // callable cannot accept residual named args
      ImmutableList<Argument> residualNamedArgs =
          residualNamed.stream().map(i -> call.getArguments().get(i)).collect(toImmutableList());
      errorf(
          residualNamedArgs.getFirst(),
          "'%s()' got unexpected keyword argument%s: %s%s",
          call.getFunction(),
          plural(residualNamedArgs.size()),
          residualNamedArgs.stream().map(Argument::getName).collect(joining(", ")),
          // If there are multiple residual named args, it's likely due to calling the wrong
          // function or misunderstanding the API, so arg spelling suggestions would not help.
          residualNamedArgs.size() == 1
              ? SpellChecker.didYouMean(
                  residualNamedArgs.getFirst().getName(),
                  callable
                      .getParameterNames()
                      .subList(
                          callable.getNumPositionalOnlyParameters(),
                          callable.getParameterNames().size()))
              : "");
      return false;
    } else {
      // residual named args go into callable's kwargs
      for (int argIndex : residualNamed) {
        Argument arg = call.getArguments().get(argIndex);
        StarlarkType argType = argTypes.get(argIndex);
        if (!checkAssignable(
            callable.getKwargsType(), argType, call, arg, "residual keyword arguments")) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Returns true if all mandatory parameters were explicitly supplied by the call or potentially
   * supplied through *args or **kwargs. Otherwise, reports an error and returns false.
   */
  private boolean checkCallMissingMandatoryArgs(
      List<String> seenMandatoryParameters,
      boolean callHasVarargs,
      boolean callHasKwargs,
      CallExpression call,
      Types.CallableType callable) {
    if (seenMandatoryParameters.size() < callable.getMandatoryParameters().size()) {
      ImmutableSet<String> seenMandatorySet = ImmutableSet.copyOf(seenMandatoryParameters);
      // Identify mandatory parameters which were not seen and which cannot be possibly supplied
      // from the call's *args or **kwargs.
      // TODO: #28037 - Perhaps report an error if no element of varargsElementTypes /
      // kwargsValueTypes is assignable to a missing parameter's type.
      ArrayList<String> missingMandatory = new ArrayList<>(0);
      for (int i = 0; i < callable.getParameterNames().size(); i++) {
        String name = callable.getParameterNames().get(i);
        if (!callable.getMandatoryParameters().contains(name)) {
          continue;
        }
        if (!seenMandatorySet.contains(name)) {
          if (i < callable.getNumPositionalOnlyParameters() && !callHasVarargs) {
            missingMandatory.add(name);
          } else if (i < callable.getNumPositionalParameters()
              && !callHasVarargs
              && !callHasKwargs) {
            missingMandatory.add(name);
          } else if (i >= callable.getNumPositionalParameters() && !callHasKwargs) {
            missingMandatory.add(name);
          }
        }
      }
      if (!missingMandatory.isEmpty()) {
        errorf(
            call.getLparenLocation(),
            "'%s()' missing %d required argument%s: %s",
            call.getFunction(),
            missingMandatory.size(),
            plural(missingMandatory.size()),
            Joiner.on(", ").join(missingMandatory));
        return false;
      }
    }
    return true;
  }

  private StarlarkType inferComprehension(Comprehension comp) {
    for (Comprehension.Clause clause : comp.getClauses()) {
      switch (clause) {
        case Comprehension.For forClause -> {
          checkForClause(
              forClause.getVars(), forClause.getIterable(), "comprehension 'for' clause");
        }
        case Comprehension.If ifClause -> {
          // Infer only to type-check. Condition is evaluated as truthy/falsy, which is valid for
          // every type.
          var unused = infer(ifClause.getCondition());
        }
      }
    }
    if (comp.isDict()) {
      DictExpression.Entry bodyEntry = (DictExpression.Entry) comp.getBody();
      return Types.dict(infer(bodyEntry.getKey()), infer(bodyEntry.getValue()));
    } else {
      Expression bodyElement = (Expression) comp.getBody();
      return Types.list(infer(bodyElement));
    }
  }

  /** Recursively type-checks the vars and the iterable, and assigns the vars to the iterable. */
  private void checkForClause(Expression vars, Expression iterable, String what) {
    StarlarkType iterableType = infer(iterable);
    StarlarkType varsRhsType; // The type of the value assigned to the vars expression.
    if (iterableType.equals(Types.ANY)) {
      varsRhsType = Types.ANY;
    } else {
      ArrayList<StarlarkType> varUnionElements = new ArrayList<>();
      for (StarlarkType iterableUnionElement : Types.unfoldUnion(iterableType)) {
        // TODO: #28037 - Replace with inferring T when assigning iterableType to Collection[T]
        // TODO: #28037 - Introduce an Iterable type and use it here to match language spec.
        if (iterableUnionElement.equals(Types.ANY)) {
          varUnionElements.add(Types.ANY);
        } else if (iterableUnionElement instanceof Types.AbstractCollectionType collection) {
          varUnionElements.add(collection.getElementType());
        } else {
          errorf(iterable, "%s operand must be an iterable, got '%s'", what, iterableType);
        }
      }
      varsRhsType = Types.union(varUnionElements);
    }
    assign(vars, varsRhsType);
  }

  private boolean checkAssignable(
      @Nullable StarlarkType lhs,
      @Nullable StarlarkType rhs,
      CallExpression call,
      Node node,
      String nodeDescription) {
    if (lhs != null && rhs != null) {
      if (!StarlarkType.assignableFrom(lhs, rhs)) {
        errorf(
            node,
            "in call to '%s()', %s must be '%s', not '%s'",
            call.getFunction(),
            nodeDescription,
            lhs,
            rhs);
        return false;
      }
    }
    return true;
  }

  private static StarlarkType inferTupleRepetition(Types.TupleType tuple, Expression timesExpr) {
    @Nullable Integer times = getIntValueExact(timesExpr);
    if (times != null) {
      return tuple.repeat(times);
    }
    return tuple.toHomogeneous();
  }

  /**
   * Returns the inferred type of an expression.
   *
   * <p>The expression must have already been resolved and successfully type-tagged, i.e.
   * identifiers must have their bindings set and these bindings must contain type information.
   *
   * @throws SyntaxError.Exception if a static type error is present in the expression.
   */
  static StarlarkType inferTypeOf(Expression expr, TypeTable typeTable, TypeContext typeContext)
      throws SyntaxError.Exception {
    TypeChecker tc = new TypeChecker(typeTable, typeContext);
    StarlarkType result = tc.infer(expr);
    if (!typeTable.ok()) {
      throw new SyntaxError.Exception(typeTable.errors());
    }
    return result;
  }

  /**
   * Recursively typechecks the assignment of type {@code rhsType} to the target expression {@code
   * lhs}.
   *
   * <p>Mutates the types on the {@link Resolver.Binding} objects of untyped variables by setting
   * them to their inferred type (if this is the first assignment to that variable in typed code).
   *
   * <p>The asymmetry of the parameter types comes from the fact that this helper recursively
   * decomposes the LHS syntactically, whereas the RHS has already been fully evaluated to a type.
   * For instance, {@code x, y = (1, 2)} and {@code x, y = my_pair} both trigger the same behavior
   * in this method. Decomposing the LHS syntactically rather than by type is what allows {@code (x,
   * y) = [1, 2]} to succeed, even though assignment of a list to a tuple type is illegal (as in
   * {@code t : Tuple[int, int] = [1, 2]}).
   */
  private void assign(Expression lhs, StarlarkType rhsType) {
    checkState(usesTypeSyntax());

    if (lhs.kind() == Expression.Kind.LIST_EXPR) {
      assignSequence((ListExpression) lhs, rhsType);
      return;
    }

    ImmutableList<StarlarkType> lhsMeet = inferIndividualAssignmentTarget(lhs);
    for (StarlarkType lhsType : lhsMeet) {
      if (!StarlarkType.assignableFrom(lhsType, rhsType)) {
        errorf(lhs, "cannot assign type '%s' to %s", rhsType, formatExprWithMeetType(lhs, lhsMeet));
        break;
      }
    }

    if (lhs instanceof Identifier id && typeTable.getType(id.getBinding()) == null) {
      // If a variable has not been typed, infer its type from the rhs of the first assignment.
      typeTable.setInferredType(id.getBinding(), rhsType.toLvalue());
    }
  }

  private static String formatExprWithMeetType(Expression expr, ImmutableList<StarlarkType> types) {
    if (types.size() == 1) {
      return String.format("'%s' of type '%s'", expr, types.getFirst());
    } else {
      return String.format(
          "'%s' which expects a value satisfying all of the %d types [%s]",
          expr,
          types.size(),
          types.stream().map(t -> String.format("'%s'", t)).collect(joining(", ")));
    }
  }

  /**
   * Verifies that the expression can be used as the target of a non-sequence assignment (or
   * augmented assignment). Returns a non-flattened unfolded list of LHS acceptor types, each of
   * which must be checked for being assignable by the assignment's RHS type.
   *
   * <p>In type theory terms, the returned list represents the meet of its type elements; however,
   * meet types don't (yet) exist in the Starlark type system.
   *
   * <p>If the LHS is an index or dot expression whose object is of a union type, then each of the
   * possible acceptor types must be assignable. We want to distinguish between the valid case
   * {@code x: list[int|str]; x[0] = 1} (where there is a single LHS acceptor type, int|str) and the
   * invalid case {@code y: list[int] | list[str]; y[0] = 1} (which has a pair of LHS acceptor
   * types, int and str, the latter of which is not assignable from 1).
   */
  private ImmutableList<StarlarkType> inferIndividualAssignmentTarget(Expression lhs) {
    switch (lhs.kind()) {
      case Expression.Kind.INDEX -> {
        IndexExpression indexExpr = (IndexExpression) lhs;
        StarlarkType objectType = infer(indexExpr.getObject());
        StarlarkType keyType = infer(indexExpr.getKey());
        if (!objectType.hasSetIndex()) {
          errorf(
              lhs,
              "%s of type '%s' does not support item assignment",
              indexExpr.getObject(),
              objectType);
        }
        return inferIndexUnfolded(indexExpr, objectType, keyType);
      }
      case Expression.Kind.DOT -> {
        DotExpression dotExpr = (DotExpression) lhs;
        StarlarkType objectType = infer(dotExpr.getObject());
        if (!objectType.hasSetField()) {
          errorf(
              lhs,
              "%s of type '%s' does not support field assignment",
              dotExpr.getObject(),
              objectType);
        }
        return inferDotUnfolded(dotExpr, objectType);
      }
      case Expression.Kind.IDENTIFIER -> {
        return ImmutableList.of(infer(lhs));
      }
      default -> {
        StarlarkType lhsType = infer(lhs);
        errorf(lhs, "%s of type '%s' is not a valid target for assignment", lhs, lhsType);
        return ImmutableList.of(Types.ANY);
      }
    }
  }

  private void assignSequence(ListExpression lhs, StarlarkType rhsType) {
    if (rhsType.equals(Types.ANY)) {
      for (Expression element : lhs.getElements()) {
        assign(element, Types.ANY);
      }
      return;
    }

    // We effectively need to transform what may be a union of iterables into a fixed-length tuple
    // of unions; e.g. list[int] | tuple[str, bool] => tuple[int | str, int | bool].
    // (Of course, any tuples in the rhsType union must be of the expected length.)
    ImmutableCollection<StarlarkType> rhsUnionElements = Types.unfoldUnion(rhsType);
    for (StarlarkType rhsUnionElement : rhsUnionElements) {
      if (rhsUnionElement instanceof Types.FixedLengthTupleType rhsTuple) {
        if (lhs.getElements().size() != rhsTuple.getElementTypes().size()) {
          errorf(
              lhs,
              "cannot assign type '%s' to '%s'; want %d-element sequence",
              rhsType,
              lhs,
              lhs.getElements().size());
          return;
        }
      } else if (!Types.isCollection(rhsType)) {
        // TODO: #28043 - consider checking for an Iterable type (as it is in the eval layer)
        errorf(lhs, "cannot assign non-iterable type '%s' to '%s'", rhsType, lhs);
        return;
      }
    }
    for (int i = 0; i < lhs.getElements().size(); i++) {
      ArrayList<StarlarkType> rhsElementTypes = new ArrayList<>(rhsUnionElements.size());
      for (StarlarkType rhsUnionElement : rhsUnionElements) {
        if (rhsUnionElement instanceof Types.FixedLengthTupleType rhsTuple) {
          rhsElementTypes.add(rhsTuple.getElementTypes().get(i));
        } else if (rhsUnionElement instanceof Types.AbstractCollectionType rhsCollection) {
          rhsElementTypes.add(rhsCollection.getElementType());
        } else if (rhsUnionElement.equals(Types.ANY)) {
          rhsElementTypes.add(Types.ANY);
        }
      }
      assign(lhs.getElements().get(i), Types.union(rhsElementTypes));
    }
  }

  private void visitProgram(Program prog) {
    checkState(
        functionStack.isEmpty(),
        "When type-checkings a Program, functionStack is expected to be initially empty");
    Resolver.Function toplevel = prog.getResolvedFunction();
    this.functionStack.push(toplevel);
    visitBlock(toplevel.getBody());
    checkState(functionStack.pop().equals(toplevel));
  }

  @Override
  public void visit(StarlarkFile file) {
    checkState(
        functionStack.isEmpty(),
        "When type-checkings a StarlarkFile, functionStack is expected to be initially empty");
    Resolver.Function toplevel = file.getResolvedFunction();
    this.functionStack.push(toplevel);
    super.visit(file);
    checkState(functionStack.pop().equals(toplevel));
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
    if (!usesTypeSyntax()) {
      return;
    }

    if (assignment.isAugmented()) {
      TokenKind operator = assignment.getOperator();
      Location operatorLocation = assignment.getOperatorLocation();
      Expression lhs = assignment.getLHS();
      Expression rhs = assignment.getRHS();
      ImmutableList<StarlarkType> lhsMeet = inferIndividualAssignmentTarget(lhs);
      StarlarkType rhsType = infer(assignment.getRHS());
      for (StarlarkType lhsType : lhsMeet) {
        // TODO(b/141263526): if we decide to support list += sequence, we'd need to special-case it
        // here (since list + tuple is an error per inferBinaryOperator()).
        StarlarkType resultType =
            inferBinaryOperator(
                lhs,
                lhsType,
                operator,
                operatorLocation,
                rhs,
                rhsType,
                /* augmentedAssignment= */ true);
        if (!StarlarkType.assignableFrom(lhsType, resultType)) {
          binaryOperatorError(
              lhsType,
              operator,
              operatorLocation,
              rhsType,
              /* augmentedAssignment= */ true,
              String.format(
                  "cannot update %s with a result value of type '%s'",
                  formatExprWithMeetType(lhs, lhsMeet), resultType));
        }
      }
    } else {

      var rhsType = infer(assignment.getRHS());

      assign(assignment.getLHS(), rhsType);
    }
  }

  @Override
  public void visit(ForStatement node) {
    if (usesTypeSyntax()) {
      checkForClause(node.getVars(), node.getCollection(), "'for' loop");
    }
    // Visit the for loop body even in untyped code; it may contain nested typed def statements.
    visitBlock(node.getBody());
  }

  @Override
  public void visit(DefStatement def) {
    Resolver.Function function = def.getResolvedFunction();
    functionStack.push(function);
    if (typeTable.usesTypeSyntax(function)) {
      Types.CallableType callableType =
          checkNotNull(
              typeTable.getType(function),
              "type tagger should have set type for def statement '%s'",
              def);
      int numOrdinaryParams = callableType.getParameterTypes().size();
      for (int i = 0; i < numOrdinaryParams; i++) {
        Parameter param = def.getParameters().get(i);
        if (param.getDefaultValue() != null) {
          StarlarkType defaultValueType = infer(param.getDefaultValue());
          if (!StarlarkType.assignableFrom(
              callableType.getParameterTypeByPos(i), defaultValueType)) {
            errorf(
                param.getDefaultValue().getStartLocation(),
                "%s(): parameter '%s' has default value of type '%s', declares '%s'",
                def.getIdentifier().getName(),
                param.getName(),
                defaultValueType,
                callableType.getParameterTypeByPos(i));
          }
        }
      }

      @Nullable Statement implicitNoneReturn = getImplicitNoneReturn(def.getBody());
      if (implicitNoneReturn != null
          && !StarlarkType.assignableFrom(callableType.getReturnType(), Types.NONE)) {
        errorf(
            implicitNoneReturn,
            "%s() declares return type '%s' but may exit without an explicit 'return'",
            def.getIdentifier().getName(),
            callableType.getReturnType());
      }
    }

    // Visit body even in untyped code; it may contain nested typed def statements.
    visitBlock(def.getBody());
    checkState(functionStack.poll() == function);
  }

  @Override
  public void visit(IfStatement node) {
    if (usesTypeSyntax()) {
      // Check type constraints in the condition.
      infer(node.getCondition());
    }
    // Visit then/else blocks even in untyped code; they may contain nested typed def statements.
    visitBlock(node.getThenBlock());
    if (node.getElseBlock() != null) {
      visitBlock(node.getElseBlock());
    }
  }

  @Override
  public void visit(ExpressionStatement expr) {
    if (!usesTypeSyntax()) {
      return;
    }
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
  public void visit(ReturnStatement ret) {
    if (!usesTypeSyntax()) {
      return;
    }
    StarlarkType returnType = ret.getResult() == null ? Types.NONE : infer(ret.getResult());
    checkState(!functionStack.isEmpty());
    Resolver.Function function = functionStack.peek();
    // May be null if function is the toplevel
    @Nullable Types.CallableType callableType = typeTable.getType(function);
    if (callableType != null
        && !StarlarkType.assignableFrom(callableType.getReturnType(), returnType)) {
      errorf(
          ret.getResult().getStartLocation(),
          "%s() declares return type '%s' but may return '%s'",
          function.getName(),
          callableType.getReturnType(),
          returnType);
    }
  }

  @Override
  public void visit(TypeAliasStatement alias) {
    // Don't descend into children.
  }

  @Override
  public void visit(VarStatement var) {
    // Don't descend into children.
  }

  /**
   * Heuristically checks whether a function body ends with an implicit {@code None} return, i.e. a
   * non-return statement, and if so, retrieves the statement after which the implicit {@code None}
   * return occurs. Recurses into if statement bodies.
   *
   * <p>This check doesn't attempt to detect unreachable code within the body, so e.g.
   *
   * <pre>
   * def f() -> int:
   *     return 1
   *     pass
   * </pre>
   *
   * will be flagged as implicitly returning {@code None} on the unreachable last line.
   *
   * @return the first statement after which the function exits and the implicit {@code None} return
   *     occurs, or {@code null} if none was found
   */
  @Nullable
  private static Statement getImplicitNoneReturn(ImmutableList<Statement> body) {
    Statement last = body.getLast();
    if (last instanceof ReturnStatement) {
      return null;
    } else if (last instanceof IfStatement ifStmt) {
      // An if statement is considered to have an explicit return if it has both `then` and `else`
      // branches, and both branches end with an explicit return.
      if (ifStmt.getElseBlock() == null) {
        return ifStmt;
      }
      @Nullable Statement thenImplicitNoneReturn = getImplicitNoneReturn(ifStmt.getThenBlock());
      return thenImplicitNoneReturn != null
          ? thenImplicitNoneReturn
          : getImplicitNoneReturn(ifStmt.getElseBlock());
    }
    return last;
  }

  /**
   * Returns true if the current function is considered to use type syntax, or if we were invoked
   * via {@link #inferTypeOf}. If false, the current node must not be type-checked.
   */
  private boolean usesTypeSyntax() {
    return functionStack.isEmpty() || typeTable.usesTypeSyntax(functionStack.peek());
  }

  private static void checkFileOptions(FileOptions options) {
    checkArgument(
        options.resolveTypeSyntax(), "static type checking requires that resolveTypeSyntax is set");
    checkArgument(
        !options.tolerateInvalidTypeExpressions(),
        "static type checking requires that tolerateInvalidTypeExpressions is not set");
  }

  /**
   * Checks that the given file's AST satisfies the types in the bindings of its identifiers.
   *
   * <p>The file must have already been passed through the {@link TypeTagger} without error
   *
   * <p>Any type checking errors are appended to the type table's errors list.
   *
   * @throws IllegalArgumentException if the file's {@link FileOptions} don't contain {@link
   *     FileOptions#resolveTypeSyntax()} or do contain {@link
   *     FileOptions#tolerateInvalidTypeExpressions()}.
   */
  public static void checkFile(StarlarkFile file, TypeTable typeTable, TypeContext typeContext) {
    checkFileOptions(file.getOptions());
    TypeChecker checker = new TypeChecker(typeTable, typeContext);
    checker.visit(file);
  }

  /**
   * Like {@link #checkFile}, but on an already-compiled {@link Program}.
   *
   * <p>The program is *not* mutated. Any errors are appended to the type table's errors list.
   *
   * @throws IllegalArgumentException if the program's {@link FileOptions} don't contain {@link
   *     FileOptions#resolveTypeSyntax()} or do contain {@link
   *     FileOptions#tolerateInvalidTypeExpressions()}.
   */
  public static void checkProgram(Program prog, TypeTable typeTable, TypeContext typeContext) {
    checkFileOptions(prog.getOptions());
    TypeChecker checker = new TypeChecker(typeTable, typeContext);
    checker.visitProgram(prog);
  }
}
