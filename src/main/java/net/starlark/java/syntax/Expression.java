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
public abstract class Expression extends Node {

  /**
   * Kind of the expression. This is similar to using instanceof, except that it's more efficient
   * and can be used in a switch/case.
   */
  public enum Kind {
    BINARY_OPERATOR,
    CALL,
    CAST,
    COMPREHENSION,
    CONDITIONAL,
    DICT_EXPR,
    DOT,
    ELLIPSIS,
    FLOAT_LITERAL,
    IDENTIFIER,
    INDEX,
    INT_LITERAL,
    ISINSTANCE,
    LAMBDA,
    LIST_EXPR,
    SLICE,
    STRING_LITERAL,
    UNARY_OPERATOR,
    TYPE_APPLICATION,
  }

  // The ordinals of the kinds as constants, so that the interpreter can switch on kindOrdinal()
  // without loading the Kind.
  public static final int BINARY_OPERATOR = 0;
  public static final int CALL = 1;
  public static final int CAST = 2;
  public static final int COMPREHENSION = 3;
  public static final int CONDITIONAL = 4;
  public static final int DICT_EXPR = 5;
  public static final int DOT = 6;
  public static final int ELLIPSIS = 7;
  public static final int FLOAT_LITERAL = 8;
  public static final int IDENTIFIER = 9;
  public static final int INDEX = 10;
  public static final int INT_LITERAL = 11;
  public static final int ISINSTANCE = 12;
  public static final int LAMBDA = 13;
  public static final int LIST_EXPR = 14;
  public static final int SLICE = 15;
  public static final int STRING_LITERAL = 16;
  public static final int UNARY_OPERATOR = 17;
  public static final int TYPE_APPLICATION = 18;

  private static final Kind[] KINDS = Kind.values();

  // The kind and an optional offset, packed by packKindAndOffset.
  private final int kindAndOffset;

  Expression(FileLocations locs, Kind kind) {
    super(locs);
    this.kindAndOffset = kind.ordinal();
  }

  /** Also stores an offset that subclasses can retrieve with {@link #packedOffset}. */
  Expression(FileLocations locs, Kind kind, int offset) {
    super(locs);
    this.kindAndOffset = packKindAndOffset(kind, offset);
  }

  /**
   * Kind of the expression. This is similar to using instanceof, except that it's more efficient
   * and can be used in a switch/case.
   */
  // Final to avoid cost of virtual call (see #12967).
  public final Kind kind() {
    return KINDS[kindOrdinal()];
  }

  /** Returns the ordinal of {@link #kind}, which equals the int constant of the same name. */
  public final int kindOrdinal() {
    return kindAndOffset & ((1 << KIND_BITS) - 1);
  }

  /** Returns the offset passed to the constructor. */
  final int packedOffset() {
    return kindAndOffset >>> KIND_BITS;
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
