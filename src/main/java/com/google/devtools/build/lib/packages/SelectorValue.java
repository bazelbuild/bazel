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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.HasBinary;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.syntax.TokenKind;
import java.util.Map;
import net.starlark.java.annot.StarlarkBuiltin;

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
@AutoCodec
public final class SelectorValue implements StarlarkValue, HasBinary {

  // TODO(adonovan): combine Selector{List,Value} and BuildType.SelectorList.
  // We don't need three classes for the same concept.

  private final ImmutableMap<?, ?> dictionary;
  private final Class<?> type;
  private final String noMatchError;

  SelectorValue(Map<?, ?> dictionary, String noMatchError) {
    Preconditions.checkArgument(!dictionary.isEmpty());
    this.dictionary = ImmutableMap.copyOf(dictionary);
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
  public SelectorList binaryOp(TokenKind op, Object that, boolean thisLeft) throws EvalException {
    if (op == TokenKind.PLUS) {
      return thisLeft ? SelectorList.concat(this, that) : SelectorList.concat(that, this);
    }
    return null;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("select(").repr(dictionary).append(")");
  }
}
