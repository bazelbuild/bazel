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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.spelling.SpellChecker;

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

  private static String plural(int n) {
    return n == 1 ? "" : "s";
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
      case CALL -> {
        return inferCall((CallExpression) expr);
      }
      case CONDITIONAL -> {
        var cond = (ConditionalExpression) expr;
        return Types.union(infer(cond.getThenCase()), infer(cond.getElseCase()));
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
        // TODO: #28037 - support cast, comprehension, lambda, and slice expressions.
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
            .map(Argument::getValue)
            .map(this::infer)
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
  public void visit(DefStatement def) {
    // TODO: #28037 - If the def statement is typed, verify default parameters and return type.
    // The current no-op version exists only for call expression testing.
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
