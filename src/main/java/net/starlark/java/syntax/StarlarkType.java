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
   *
   * <p>This method is intended to return only supertypes of other {@link StarlarkType} subclasses.
   * To express subtype/supertype relation with other instances of the same {@link StarlarkType}
   * subclass, use {@link #assignableFromHook}.
   */
  // TODO: #27370 - Add getSubtypes(), with the semantics that the actual subtype relation is the
  // union of these two methods.
  public List<StarlarkType> getSupertypes() {
    return ImmutableList.of();
  }

  /**
   * Returns true if Starlark values of this type can be assigned from Starlark values of type
   * {@code t}, where {@code t} is not {@code Any}, {@code Object}, {@code Never}, or a union; and
   * not a type one of whose {@link #getSupertypes} is assignable to this type.
   *
   * <p>Subclasses of {@link StarlarkType} should override this method to implement covariance,
   * contravariance, consistent-equals-based invariance, or rvalue assignability rules in type
   * arguments.
   *
   * <p>See {@link #getSupertypes()} for subtype/supertype relation with other (non-{@link
   * #toRvalue}) subclasses of {@link StarlarkType}.
   */
  public boolean assignableFromHook(StarlarkType t) {
    return this.equals(t);
  }

  /**
   * If this type has a field by the given name, returns the type of that field, or null otherwise.
   */
  @Nullable
  public StarlarkType getField(String name, TypeContext context) {
    return null;
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
  public static boolean assignableFrom(StarlarkType t1, StarlarkType t2) {
    if (t1.equals(Types.ANY) || t2.equals(Types.ANY)) {
      return true;
    }
    if (t1.equals(Types.OBJECT)) {
      return true;
    }
    if (t2.equals(Types.NEVER)) {
      return true;
    }
    if (t1.equals(t2)) {
      return true;
    }
    if (t1.assignableFromHook(t2)) {
      return true;
    }
    if (t2 instanceof Types.UnionType union2) {
      return union2.getTypes().stream().allMatch(sub2 -> assignableFrom(t1, sub2));
    }
    if (t1 instanceof Types.UnionType union1) {
      return union1.getTypes().stream().anyMatch(sub1 -> assignableFrom(sub1, t2));
    }
    if (t2.getSupertypes().stream().anyMatch(super2 -> assignableFrom(t1, super2))) {
      return true;
    }
    return false;
  }

  /**
   * Returns true if values of the two types can be assigned to each other.
   *
   * <p>This relationship is symmetric but not transitive; {@code int} and {@code float} are both
   * consistent-equals to {@code Any}, but {@code int} and {@code float} are not consistent-equals
   * to each other.
   *
   * <p>The consistent-equals check should be used for type parameter invariance; for example,
   * {@code list[int]} is consistent-equals to {@code list[Any]} because {@code int} is
   * consistent-equals to {@code Any}.
   */
  public static boolean consistentEquals(StarlarkType t1, StarlarkType t2) {
    return assignableFrom(t1, t2) && assignableFrom(t2, t1);
  }

  /**
   * If this is (or contains, in the case of a composite type) an "Rvalue type", returns the
   * corresponding non-Rvalue type to use when this type is propagated to a variable, parameter, or
   * function return. Otherwise, if this is a non-Rvalue type, returns this type itself.
   *
   * <p>We use Rvalue types for expressions that evaluate to a new, unaliased value. For instance,
   * the Rvalue list type is used for list literal expressions and for the list concatenation /
   * multiplication binary operations. (In type theory it seems like this is a <a
   * href="https://en.wikipedia.org/wiki/Uniqueness_type">Uniqueness Type</a>.) When a Rvalue-typed
   * expression is aliased by assigning to a variable, or passing it to or returning it from a
   * function, is is replaced by the corresponding non-Rvalue type (list in this example).
   *
   * <p>The point of an Rvalue type is that, since it cannot be aliased, it allows a mutable type to
   * safely have covariance/contravariance, whereas the non-Rvalue type must be invariant. To see
   * why mutable types cannot be covariant, consider the code:
   *
   * {@snippet :
   *     x : list[int] = ...
   *     y : list[object] = x  # allowed if lists were covariant
   *     y.append("abc")       # modifies x
   *     print(sum(x))         # dynamic error due to violating type safety
   * }
   *
   * <p>Yet if all lists were invariant, that would prohibit obviously safe assignments such as:
   *
   * {@snippet :
   *     x : list[int] = []               # can't convert from list[Never]
   *     y : list[int|float] = [1, 2, 3]  # can't convert from list[int]
   * }
   *
   * <p>This problem is solved, without the need for casts, by inferring the list literals to have
   * an rvalue-list type, covariantly promoting it to the rvalue version of the LHS's list type, and
   * finally changing it to non-Rvalue type as returned by this method.
   *
   * <p>Invariant: for any {@code T} which is assignable to {@code U}, {@code T} is also assignable
   * to {@code U.toLvalue()}.
   *
   * <p>Composite types (e.g. collections and unions) must override this method and implement it
   * recursively.
   */
  // TODO: #27370 - when we have generics and type deconstruction, we should have a static method
  // performing recursive deconstruction and lvalue-ification rather than relying on StarlarkType
  // subclasses' overrides.
  public StarlarkType toLvalue() {
    return this;
  }

  /**
   * Returns the type that should be inferred for a new, unaliased value of this type; for example,
   * a literal, the result of a binary operation, or the return value of a constructor function.
   *
   * <p>See {@link #toLvalue} for further discussion.
   *
   * <p>Invariant: for any {@code U} which is assignable from {@code T}, {@code U} is also
   * assignable from {@code T.toRvalue()}.
   *
   * <p>Implementations generally should not be recursive, unlike {@link #toLvalue}.
   */
  public StarlarkType toRvalue() {
    return this;
  }

  /**
   * Infers the return type of a binary operation having an operand of this type. Intended for use
   * by {@link TypeChecker}.
   *
   * @param operator a binary operator (one of {@link BinaryOperatorExpression#operators}) which is
   *     not a truthiness, equality, or ordering comparison operator (in other words, not {@link
   *     TokenKind#AND}, {@link TokenKind#OR}, {@link TokenKind#EQUALS}, {@link
   *     TokenKind#NOT_EQUALS}, {@link TokenKind#LESS}, {@link TokenKind#LESS_EQUALS}, {@link
   *     TokenKind#GREATER}, or {@link TokenKind#GREATER_EQUALS}); those are handled specially by
   *     {@link TypeChecker#infer}.
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

  /**
   * Returns true iff the values of the two arbitrary (possibly union) types can be ordering
   * compared.
   */
  public static boolean comparable(StarlarkType x, StarlarkType y) {
    return x.isComparable(y) || y.isComparable(x);
  }

  /**
   * Returns true if this type's values can be ordering compared with values of another type. A
   * return value of false is ambiguous on its own; two types are considered incomparable iff both
   * {code x.isComparable(y)} and {@code y.isComparable(x)} are false.
   *
   * <p>Do not call this method directly; instead, use {@link #comparable}.
   */
  protected boolean isComparable(StarlarkType that) {
    return false;
  }

  /**
   * Returns true if an index expression on a value of this type can be used as the LHS of an
   * assignment.
   */
  public boolean hasSetIndex() {
    return false;
  }

  /**
   * Returns true if a dot expressions on a value of this type can be used as the LHS of an
   * assignment.
   */
  public boolean hasSetField() {
    return false;
  }
}
