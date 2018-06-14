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
package com.google.devtools.build.lib.query2;

import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.query2.engine.AllRdepsFunction;
import com.google.devtools.build.lib.query2.engine.FunctionExpression;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionMapper;
import com.google.devtools.build.lib.query2.engine.RdepsFunction;
import com.google.devtools.build.lib.query2.engine.TargetLiteral;
import java.util.List;

/**
 * A {@link QueryExpressionMapper} that transforms each occurrence of an expression of the form
 * {@literal 'rdeps(<universeScope>, <T>)'} to {@literal 'allrdeps(<T>)'}. The latter is more
 * efficient.
 */
class RdepsToAllRdepsQueryExpressionMapper extends QueryExpressionMapper<Void> {
  private final TargetPattern.Parser targetPatternParser;
  private final String absoluteUniverseScopePattern;

  RdepsToAllRdepsQueryExpressionMapper(
      TargetPattern.Parser targetPatternParser, String universeScopePattern) {
    this.targetPatternParser = targetPatternParser;
    this.absoluteUniverseScopePattern = targetPatternParser.absolutize(universeScopePattern);
  }

  @Override
  public QueryExpression visit(FunctionExpression functionExpression, Void context) {
    if (functionExpression.getFunction().getName().equals(new RdepsFunction().getName())) {
      List<Argument> args = functionExpression.getArgs();
      QueryExpression universeExpression = args.get(0).getExpression();
      if (universeExpression instanceof TargetLiteral) {
        TargetLiteral literalUniverseExpression = (TargetLiteral) universeExpression;
        String absolutizedUniverseExpression =
            targetPatternParser.absolutize(literalUniverseExpression.getPattern());
        if (absolutizedUniverseExpression.equals(absoluteUniverseScopePattern)) {
          List<Argument> argsTail = args.subList(1, functionExpression.getArgs().size());
          return new FunctionExpression(new AllRdepsFunction(), argsTail);
        }
      }
    }
    return super.visit(functionExpression, context);
  }
}
