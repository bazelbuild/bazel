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
package com.google.devtools.build.lib.packages;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import java.util.ArrayList;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.HasBinary;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.TypeConstructor;
import net.starlark.java.syntax.TypeContext;
import net.starlark.java.syntax.Types;

/**
 * The value returned by a call to {@code select({...})}, for example:
 *
 * <pre>
 *   rule(
 *       name = 'myrule',
 *       deps = select({
 *           'a': [':adep'],
 *           'b': [':bdep'],
 *       })
 * </pre>
 */
@StarlarkBuiltin(
    name = "selector",
    doc = "A selector between configuration-dependent values.",
    documented = false)
public final class SelectorValue implements StarlarkValue, HasBinary {

  public static final TypeConstructor TYPE_CONSTRUCTOR =
      Types.wrapTypeConstructor("select", SelectorValue.Type::of);

  // TODO(adonovan): combine Selector{List,Value} and BuildType.SelectorList.
  // We don't need three classes for the same concept.

  private final ImmutableMap<?, ?> dictionary;
  private final Class<?> type;
  private final String noMatchError;

  SelectorValue(ImmutableMap<?, ?> dictionary, String noMatchError) {
    Preconditions.checkArgument(!dictionary.isEmpty());
    this.dictionary = dictionary;
    // TODO(adonovan): doesn't this assume all the elements have the same type?
    this.type = Iterables.getFirst(dictionary.values(), null).getClass();
    this.noMatchError = noMatchError;
  }

  ImmutableMap<?, ?> getDictionary() {
    return dictionary;
  }

  Class<?> getType() {
    return type;
  }

  /**
   * Returns a custom error message for this select when no condition matches, or an empty string if
   * no such message is declared.
   */
  String getNoMatchError() {
    return noMatchError;
  }

  @Override
  public String toString() {
    return Starlark.repr(this, StarlarkSemantics.DEFAULT);
  }

  @Override
  @Nullable
  public SelectorList binaryOp(TokenKind op, Object that, boolean thisLeft) throws EvalException {
    return SelectorList.of(this).binaryOp(op, that, thisLeft);
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("select(").repr(dictionary, semantics).append(")");
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof SelectorValue that)) {
      return false;
    }
    // TODO(bazel-team): We probably have some inconsistencies here. 1) We're not checking the
    // order of the dictionary, which is relevant to matching semantics. 2) We're checking the
    // type, which depends on the concrete type of the first entry's value, which could be a
    // subtype that is not semantically meaningful to the user. These problems are probably best
    // solved by merging this class into the BuildType-land equivalent, with normalization that
    // removes subtype distinctions by copying into standard attribute types.
    return Objects.equal(dictionary, that.dictionary)
        && Objects.equal(type, that.type)
        && Objects.equal(noMatchError, that.noMatchError);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(dictionary, type, noMatchError);
  }

  @Override
  public SelectorValue.Type getStarlarkType(StarlarkSemantics semantics) {
    return Type.of(
        Types.union(
            dictionary.values().stream()
                .map(
                    v ->
                        switch (v) {
                          // None values in selects are special; they represent the attribute's
                          // default value. Returning the bottom type seems the most reasonable
                          // choice.
                          case NoneType none -> Types.NEVER;
                          // Assume that values of a selector type cannot be mutated in reasonable
                          // usage (although Bazel technically allows it:
                          // https://github.com/bazelbuild/bazel/issues/30094),
                          // so that empty lists/dicts are of Never rather than of Any. (This gives
                          // us the covariant behavior we want when using a load()-ed select.)
                          case StarlarkList<?> list when list.isEmpty() -> Types.list(Types.NEVER);
                          case Dict<?, ?> dict when dict.isEmpty() ->
                              Types.dict(Types.NEVER, Types.NEVER);
                          default -> Starlark.getStarlarkType(v, semantics);
                        })
                .collect(toImmutableSet())));
  }

  /** The Starlark type of {@code select} expressions. */
  @AutoValue
  public abstract static class Type extends StarlarkType {
    public static Type of(StarlarkType valueType) {
      // Assume that values of a selector type cannot be mutated in reasonable usage (although Bazel
      // technically allows it: https://github.com/bazelbuild/bazel/issues/30094), so we can treat
      // them as rvalues. This, in particular, allows us to make the (dynamic) type of
      // `select({"//foo": [], "//bar": ["something"]})}) assignable to a `select[list[str]]`,
      // rather than only to a `select[list[Any]|list[str]]`.
      return new AutoValue_SelectorValue_Type(valueType.toRvalue());
    }

    public abstract StarlarkType getValueType();

    @Override
    public final String toString() {
      StarlarkType valueType = getValueType();
      return valueType.equals(Types.ANY) ? "select" : String.format("select[%s]", valueType);
    }

    @Override
    public boolean assignableFromHook(StarlarkType t, TypeContext context) {
      if (t instanceof SelectorValue.Type thatSelector) {
        return StarlarkType.assignableFrom(getValueType(), thatSelector.getValueType(), context);
      }
      return false;
    }

    @Override
    @Nullable
    public StarlarkType inferBinaryOperator(
        TokenKind operator, StarlarkType that, boolean thisLeft) {
      // The inferred type is a select of the type of the operator applied to the value types of the
      // operands (if selector types) or the operand types themselves (otherwise). The main
      // complication is that we need to unfold unions ourselves, replicating part of the logic of
      // TypeChecker.inferBinaryOperator().
      if (operator != TokenKind.PLUS && operator != TokenKind.PIPE) {
        return null;
      }
      if (that instanceof SelectorValue.Type && !thisLeft) {
        // Avoid repeating the work if both sides are selector types.
        return null;
      }
      ImmutableCollection<StarlarkType> thisValueTypes = Types.unfoldUnion(getValueType());
      ImmutableCollection<StarlarkType> thatValueTypes =
          Types.unfoldUnion(
              that instanceof SelectorValue.Type thatSelector ? thatSelector.getValueType() : that);
      ArrayList<StarlarkType> resultTypes = new ArrayList<>();
      for (StarlarkType thisValueType : thisValueTypes) {
        for (StarlarkType thatValueType : thatValueTypes) {
          StarlarkType lhsValueType = thisLeft ? thisValueType : thatValueType;
          StarlarkType rhsValueType = thisLeft ? thatValueType : thisValueType;
          StarlarkType result =
              StarlarkType.inferBinaryOperator(lhsValueType, operator, rhsValueType);
          if (result != null) {
            resultTypes.add(result);
          } else {
            return null;
          }
        }
      }
      return SelectorValue.Type.of(Types.union(resultTypes));
    }
  }
}
