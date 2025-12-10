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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Resolver.Module;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;

/**
 * A visitor for annotating a resolved file with type information.
 *
 * <p>This populates the function type on the {@link Resolver.Function} objects in the AST.
 */
public class TypeResolver extends NodeVisitor {

  // TODO: #27728 - Will be used when we support non-universal type symbols.
  private final Module module;

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

  private TypeResolver(List<SyntaxError> errors, Module module) {
    this.errors = errors;
    this.module = module;
  }

  private Object evalTypeOrArg(Expression expr) {
    switch (expr.kind()) {
      case BINARY_OPERATOR:
        // Syntax sugar for union types, i.e. a|b == Union[a,b]
        BinaryOperatorExpression binop = (BinaryOperatorExpression) expr;
        if (binop.getOperator() == TokenKind.PIPE) {
          StarlarkType x = evalType(binop.getX());
          StarlarkType y = evalType(binop.getY());
          return Types.union(x, y);
        }
        errorf(expr, "binary operator '%s' is not supported", binop.getOperator());
        return Types.ANY;
      case TYPE_APPLICATION:
        TypeApplication app = (TypeApplication) expr;

        Object constructorObject = Types.TYPE_UNIVERSE.get(app.getConstructor().getName());
        if (constructorObject == null) {
          // TODO(ilist@): include possible candidates in the error message
          errorf(expr, "type constructor '%s' is not defined", app.getConstructor().getName());
          return Types.ANY;
        }
        if (!(constructorObject instanceof Types.TypeConstructorProxy constructor)) {
          errorf(
              expr,
              "'%s' is not a type constructor, cannot be applied to '%s'",
              app.getConstructor().getName(),
              app.getArguments());
          return Types.ANY;
        }
        ImmutableList<Object> arguments =
            app.getArguments().stream().map(arg -> evalTypeOrArg(arg)).collect(toImmutableList());

        try {
          return constructor.invoke(arguments);
        } catch (IllegalArgumentException e) {
          errorf(expr, "%s", e.getMessage());
          return Types.ANY;
        }
      case IDENTIFIER:
        Identifier id = (Identifier) expr;
        // TODO(ilist@): consider moving resolution/TYPE_UNIVERSE into Module interface
        // TODO: #27728 - Don't lookup in the type universe based on the identifier's name. Instead,
        // retrieve it from the Module using the Binding in the Identifier. I.e., make type
        // resolution build upon symbol resolution.
        Object result = Types.TYPE_UNIVERSE.get(id.getName());
        if (result == null) {
          // TODO(ilist@): include possible candidates in the error message
          errorf(expr, "type '%s' is not defined", id.getName());
          return Types.ANY;
        }
        return result;
      default:
        // TODO(ilist@): full evaluation: lists and dicts
        errorf(expr, "unexpected expression '%s'", expr);
        return Types.ANY;
    }
  }

  private StarlarkType evalType(Expression expr) {
    Object typeOrArg = evalTypeOrArg(expr);
    if (!(typeOrArg instanceof StarlarkType type)) {
      if (typeOrArg instanceof Types.TypeConstructorProxy) {
        errorf(expr, "expected type arguments after the type constructor '%s'", expr);
      } else {
        errorf(expr, "expression '%s' is not a valid type.", expr);
      }
      return Types.ANY;
    }
    return type;
  }

  private Types.CallableType createFunctionType(
      ImmutableList<Parameter> parameters, @Nullable Expression returnTypeExpr) {
    ImmutableList.Builder<String> names = ImmutableList.builder();
    ImmutableList.Builder<StarlarkType> types = ImmutableList.builder();
    ImmutableSet.Builder<String> mandatoryParameters = ImmutableSet.builder();

    int nparams = parameters.size();
    int numPositionalParameters = 0;
    Parameter.Star star = null;
    Parameter.StarStar starStar = null;
    int i;
    for (i = 0; i < nparams; i++) {
      Parameter param = parameters.get(i);
      if (param instanceof Parameter.Star pstar) {
        star = pstar;
        continue;
      }
      if (param instanceof Parameter.StarStar pstarstar) {
        starStar = pstarstar;
      }
      if (star == null) {
        numPositionalParameters++;
      }

      String name = param.getName();
      Expression typeExpr = param.getType();

      names.add(name);
      types.add(typeExpr == null ? Types.ANY : evalType(typeExpr));
      if (param instanceof Parameter.Mandatory) {
        mandatoryParameters.add(name);
      }
    }

    StarlarkType varargsType = Types.NONE;
    if (star != null && star.getIdentifier() != null) {
      Expression typeExpr = star.getType();
      varargsType = typeExpr == null ? Types.ANY : evalType(typeExpr);
    }

    StarlarkType kwargsType = Types.NONE;
    if (starStar != null) {
      Expression typeExpr = starStar.getType();
      kwargsType = typeExpr == null ? Types.ANY : evalType(typeExpr);
    }

    StarlarkType returnType = Types.ANY;
    if (returnTypeExpr != null) {
      returnType = evalType(returnTypeExpr);
    }

    return Types.callable(
        names.build(),
        types.build(),
        /* numPositionalOnlyParameters= */ 0,
        numPositionalParameters,
        mandatoryParameters.build(),
        varargsType,
        kwargsType,
        returnType);
  }

  /**
   * Resolves a type expression to a {@link StarlarkType}.
   *
   * @param expr a valid type expression; for example, one produced by {@link
   *     Expression#parseTypeExpression}.
   * @throws SyntaxError.Exception if expr is not a type expression or if it could not be resolved
   *     to a type.
   */
  public static StarlarkType evalTypeExpression(Expression expr, Module module)
      throws SyntaxError.Exception {
    List<SyntaxError> errors = new ArrayList<>();
    TypeResolver r = new TypeResolver(errors, module);
    StarlarkType result = r.evalType(expr);
    if (!errors.isEmpty()) {
      throw new SyntaxError.Exception(r.errors);
    }
    return result;
  }

  @Override
  public void visit(DefStatement def) {
    Types.CallableType type = createFunctionType(def.getParameters(), def.getReturnType());
    def.getResolvedFunction().setFunctionType(type);
  }

  @Override
  public void visit(LambdaExpression lambda) {
    Types.CallableType type =
        createFunctionType(lambda.getParameters(), /* returnTypeExpr= */ null);
    lambda.getResolvedFunction().setFunctionType(type);
  }

  /**
   * Sets the Starlark types of the {@link Resolver.Function}s in the given AST (which must have
   * already been processed by {@link Resolver}), based on the supplied annotations.
   */
  // TODO: #27728 - Also set type information in `Resolver.Binding`s.
  public static void resolveFile(StarlarkFile file, Module module) throws SyntaxError.Exception {
    TypeResolver r = new TypeResolver(file.errors, module);
    r.visit(file);
    if (!r.errors.isEmpty()) {
      throw new SyntaxError.Exception(r.errors);
    }
  }
}
