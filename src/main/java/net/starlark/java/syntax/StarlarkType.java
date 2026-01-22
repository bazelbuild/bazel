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

import com.google.common.collect.ImmutableList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Base class for all Starlark types.
 *
 * <p>Starlark typing is an experimental feature under development. See the tracking issue:
 * https://github.com/bazelbuild/bazel/issues/27370
 */
public abstract non-sealed class StarlarkType implements TypeConstructor.Arg {

  /**
   * Returns the list of supertypes of this type.
   *
   * <p>Preferred order is from the most specific to the least specific supertype. But if that is
   * not possible, the order can be arbitrary.
   */
  // TODO: #27370 - Add getSubtypes(), with the semantics that the actual subtype relation is the
  // union of these two methods.
  public List<StarlarkType> getSupertypes() {
    return ImmutableList.of();
  }

  /**
   * Returns whether a value of type {@code t2} can be assigned to a value of type {@code t1}.
   *
   * <p>In gradual typing terms, {@code t2} must be a "consistent subtype of" {@code t1}. This means
   * that there is a way to substitute zero or more occurrences of {@code Any} in both terms, such
   * that {@code t2} becomes a subtype of {@code t1} in the ordinary sense.
   *
   * <p>The Python glossary uses the term "assignable [to/from]" for this relation, and
   * "materialization" to refer to the process of substituting {@code Any}.
   */
  // TODO: #28043 - Add support for:
  // - subtyping (list[int] <= Sequence[int])
  // - covariance (Sequence[int] <= Sequence[object])
  // - proper treatment of materializing Any (Sequence[int] <= Sequence[Any])
  // - transitive application of all of the above, including across unions
  public static boolean assignableFrom(StarlarkType t1, StarlarkType t2) {
    if (t1.equals(Types.ANY) || t2.equals(Types.ANY)) {
      return true;
    }
    if (t1.equals(Types.OBJECT)) {
      return true;
    }
    if (t1.equals(t2)) {
      return true;
    }
    return false;
  }

  /**
   * Infers the return type of a binary operation having an operand of this type.
   *
   * @param operator a binary operator (one of {@link BinaryOperatorExpression#operators}) which is
   *     not {@link TokenKind#AND}, {@link TokenKind#OR}, {@link TokenKind#EQUALS}, or {@link
   *     TokenKind#NOT_EQUALS} (those are always inferred to produce {@link Types#BOOL})
   * @param that a non-union, non-Never type of the other operand
   * @param thisLeft true iff this type is the type of the LHS operand.
   * @return the inferred type of the operation, or {@code null} to indicate that we could not infer
   *     a return type, in which case the caller would fall back to calling {@code
   *     inferBinaryOperator} on the other operand's type, or to special-case handling for certain
   *     operators on certain built-in types (e.g. tuple multiplication).
   */
  @Nullable
  StarlarkType inferBinaryOperator(TokenKind operator, StarlarkType that, boolean thisLeft) {
    return null;
  }

  /** Returns true if this type's values can be compared with other values of the same type. */
  boolean isComparable() {
    return false;
  }
}
