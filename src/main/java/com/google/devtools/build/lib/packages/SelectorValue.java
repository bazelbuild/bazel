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

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.HasBinary;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.TokenKind;

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
    return Starlark.repr(this);
  }

  @Override
  @Nullable
  public SelectorList binaryOp(TokenKind op, Object that, boolean thisLeft) throws EvalException {
    return SelectorList.of(this).binaryOp(op, that, thisLeft);
  }

  @Override
  public void repr(Printer printer) {
    printer.append("select(").repr(dictionary).append(")");
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
}
