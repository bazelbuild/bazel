// Copyright 2006-2015 Google Inc. All rights reserved.
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

import java.util.List;

/**
 * Base class for test cases that use eval services.
 */
public abstract class AbstractEvaluationTestCase extends AbstractParserTestCase {

  public Object eval(String input) throws Exception {
    return eval(parseExpr(input));
  }

  public Object eval(String input, Environment env) throws Exception {
    return eval(parseExpr(input), env);
  }

  public static Object eval(Expression e) throws Exception {
    return eval(e, new Environment());
  }

  public static Object eval(Expression e, Environment env) throws Exception {
    return e.eval(env);
  }

  public void exec(String input, Environment env) throws Exception {
    exec(parseStmt(input), env);
  }

  public void exec(Statement s, Environment env) throws Exception {
    s.exec(env);
  }

  public static void exec(List<Statement> li, Environment env) throws Exception {
    for (Statement stmt : li) {
      stmt.exec(env);
    }
  }
}
