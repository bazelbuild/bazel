// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;

import java.util.List;

/**
 * A label(pattern, argument) filter expression, which computes the set of subset
 * of nodes in 'argument' whose label matches the unanchored regexp 'pattern'.
 *
 * <pre>expr ::= FILTER '(' WORD ',' expr ')'</pre>
 *
 * Example patterns:
 * <pre>
 * '//third_party'      Match all targets in the //third_party/...
 *                      (equivalent to 'intersect //third_party/...)
 * '\.jar$'               Match all *.jar targets.
 * </pre>
 */
class FilterFunction extends RegexFilterExpression {
  FilterFunction() {
  }

  @Override
  public String getName() {
    return "filter";
  }

  @Override
  protected String getPattern(List<Argument> args) {
    return args.get(0).getWord();
  }

  @Override
  public int getMandatoryArguments() {
    return 2;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.WORD, ArgumentType.EXPRESSION);
  }

  @Override
  protected <T> String getFilterString(QueryEnvironment<T> env, List<Argument> args, T target) {
    return env.getAccessor().getLabel(target);
  }
}
