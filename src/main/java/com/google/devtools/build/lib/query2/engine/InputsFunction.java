// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;

/**
 * A label(pattern, argument) filter expression, which computes the set of resulting actions of
 * argument whose inputs matches the regexp 'pattern'. The pattern follows java.util.regex format.
 *
 * <pre>expr ::= INPUTS '(' WORD ',' expr ')'</pre>
 *
 * Example patterns:
 *
 * <pre>
 * '//third_party/a.java'      Match all actions whose inputs includes //third_party/a.java
 * '//third_party/.*\.js'      Match all actions whose inputs includes js files under //third_party/
 * '.*'                        Match all actions
 * '*'                         Error: invalid regex
 * </pre>
 */
public class InputsFunction extends ActionFilterFunction {
  public static final String INPUTS = "inputs";

  @Override
  public String getName() {
    return INPUTS;
  }

  @Override
  public Iterable<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.WORD, ArgumentType.EXPRESSION);
  }
}
