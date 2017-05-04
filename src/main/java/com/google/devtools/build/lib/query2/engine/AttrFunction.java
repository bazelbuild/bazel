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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import java.util.List;

/**
 * An attr(attribute, pattern, argument) filter expression, which computes the set of subset of
 * nodes in 'argument' which correspond to rules with defined attribute 'attribute' with attribute
 * value matching the unanchored regexp 'pattern'. For list attributes, the attribute value will be
 * defined as a usual List.toString() representation (using '[' as first character, ']' as last
 * character and ", " as a delimiter between multiple values). Also, all label-based attributes will
 * use fully-qualified label names instead of original value specified in the BUILD file.
 *
 * <pre>expr ::= ATTR '(' ATTRNAME ',' WORD ',' expr ')'</pre>
 *
 * Examples
 *
 * <pre>
 * attr(linkshared,1,//project/...)    find all rules under in the //project/... that
 *                                 have attribute linkshared set to 1.
 * </pre>
 */
public class AttrFunction extends RegexFilterExpression {
  AttrFunction() {
  }

  @Override
  public String getName() {
    return "attr";
  }

  @Override
  protected String getPattern(List<Argument> args) {
    return args.get(1).getWord();
  }

  @Override
  public int getMandatoryArguments() {
    return 3;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.WORD, ArgumentType.WORD, ArgumentType.EXPRESSION);
  }

  @Override
  protected <T> String getFilterString(QueryEnvironment<T> env, List<Argument> args, T target) {
    throw new IllegalStateException(
        "The 'attr' regex filter gets its match values directly from getFilterStrings");
  }

  @Override
  protected <T> Iterable<String> getFilterStrings(QueryEnvironment<T> env,
      List<Argument> args, T target) {
    if (env.getAccessor().isRule(target)) {
      return env.getAccessor().getAttrAsString(target, args.get(0).getWord());
    }
    return ImmutableList.of();
  }
}
