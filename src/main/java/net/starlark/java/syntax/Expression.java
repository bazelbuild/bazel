// Copyright 2014 The Bazel Authors. All rights reserved.
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

/**
 * Base class for all expression nodes in the AST.
 *
 * <p>The only expressions permitted on the left-hand side of an assignment (such as 'lhs=rhs' or
 * 'for lhs in expr') are identifiers, dot expressions (x.y), list expressions ([expr, ...]), tuple
 * expressions ((expr, ...)), or parenthesized variants of those. In particular and unlike Python,
 * slice expressions and starred expressions cannot appear on the LHS. TODO(bazel-team): Add support
 * for assigning to slices (e.g. a[2:6] = [3]).
 */
public abstract sealed class Expression extends Node
    permits BinaryOperatorExpression,
        CallExpression,
        CastExpression,
        Comprehension,
        ConditionalExpression,
        DictExpression,
        DotExpression,
        Ellipsis,
        FloatLiteral,
        Identifier,
        IndexExpression,
        IntLiteral,
        IsInstanceExpression,
        LambdaExpression,
        ListExpression,
        SliceExpression,
        StringLiteral,
        TypeApplication,
        UnaryOperatorExpression {

  Expression(FileLocations locs) {
    super(locs);
  }

  /** Parses an expression. */
  public static Expression parse(ParserInput input, FileOptions options)
      throws SyntaxError.Exception {
    return Parser.parseExpression(input, options);
  }

  /** Parses an expression with default options. */
  public static Expression parse(ParserInput input) throws SyntaxError.Exception {
    return parse(input, FileOptions.DEFAULT);
  }

  /**
   * Parses a type expression.
   *
   * @param options parsing options; note that {@link FileOptions#allowStarlarkTypeSyntax} doesn't
   *     need to be set - this method supports Starlark types implicitly.
   */
  public static Expression parseTypeExpression(ParserInput input, FileOptions options)
      throws SyntaxError.Exception {
    return Parser.parseTypeExpression(input, options);
  }

  /** Parses a type expression with default options. */
  public static Expression parseTypeExpression(ParserInput input) throws SyntaxError.Exception {
    return parseTypeExpression(input, FileOptions.DEFAULT);
  }
}
