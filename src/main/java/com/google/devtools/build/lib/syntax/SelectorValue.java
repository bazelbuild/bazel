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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.util.Map;

/**
 * The value passed to a select({...}) statement, e.g.:
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
@SkylarkModule(
  name = "selector",
  doc = "A selector between configuration-dependent entities.",
  documented = false
)
@AutoCodec
public final class SelectorValue implements SkylarkValue {
  // TODO(bazel-team): Selectors are currently split between .packages and .syntax . They should
  // really all be in .packages, but then we'd need to figure out a way how to extend binary
  // operators, which is a non-trivial problem.
  private final ImmutableMap<?, ?> dictionary;
  private final Class<?> type;
  private final String noMatchError;

  public SelectorValue(Map<?, ?> dictionary, String noMatchError) {
    Preconditions.checkArgument(!dictionary.isEmpty());
    this.dictionary = ImmutableMap.copyOf(dictionary);
    this.type = Iterables.get(dictionary.values(), 0).getClass();
    this.noMatchError = noMatchError;
  }

  public ImmutableMap<?, ?> getDictionary() {
    return dictionary;
  }

  Class<?> getType() {
    return type;
  }

  /**
   * Returns a custom error message for this select when no condition matches, or an empty
   * string if no such message is declared.
   */
  public String getNoMatchError() {
    return noMatchError;
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.format("select(%r)", dictionary);
  }
}
