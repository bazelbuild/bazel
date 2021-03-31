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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.query2.engine.AllRdepsFunction;
import com.google.devtools.build.lib.query2.engine.FunctionExpression;
import com.google.devtools.build.lib.query2.engine.KindFunction;
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
  private final TargetPattern absoluteUniverseScopePattern;

  RdepsToAllRdepsQueryExpressionMapper(
      TargetPattern.Parser targetPatternParser, TargetPattern absoluteUniverseScopePattern) {
    this.targetPatternParser = targetPatternParser;
    this.absoluteUniverseScopePattern = absoluteUniverseScopePattern;
  }

  @Override
  public QueryExpression visit(FunctionExpression functionExpression, Void context) {
    if (functionExpression.getFunction().getName().equals(new RdepsFunction().getName())) {
      List<Argument> args = functionExpression.getArgs();
      QueryExpression rdepsUniverseExpression = args.get(0).getExpression();
      if (rdepsUniverseExpression instanceof TargetLiteral) {
        Eligibility eligibility =
            determineEligibility(
                targetPatternParser,
                absoluteUniverseScopePattern,
                ((TargetLiteral) rdepsUniverseExpression).getPattern());
        switch (eligibility) {
          case ELIGIBLE_AS_IS:
            return new FunctionExpression(
                new AllRdepsFunction(), args.subList(1, functionExpression.getArgs().size()));
          case ELIGIBLE_WITH_FILTERING:
            return new FunctionExpression(
                new KindFunction(),
                ImmutableList.of(
                    Argument.of(" rule$"),
                    Argument.of(
                        new FunctionExpression(
                            new AllRdepsFunction(),
                            args.subList(1, functionExpression.getArgs().size())))));
          default:
            // Do nothing. The return statement at the bottom of the method is what we want.
        }
      }
    }
    return super.visit(functionExpression, context);
  }

  /**
   * Describes how eligible, if at all, a `rdeps(pattern, E, d)` expression is for being transformed
   * to one that uses `allrdeps`.
   */
  enum Eligibility {
    NOT_ELIGIBLE,

    ELIGIBLE_WITH_FILTERING,

    ELIGIBLE_AS_IS,
  }

  static Eligibility determineEligibility(
      TargetPattern.Parser targetPatternParser,
      TargetPattern absoluteUniverseScopePattern,
      String rdepsUniversePatternString) {
    TargetPattern absoluteRdepsUniverseTargetPattern;
    try {
      absoluteRdepsUniverseTargetPattern =
          targetPatternParser.parse(targetPatternParser.absolutize(rdepsUniversePatternString));
    } catch (TargetParsingException e) {
      return Eligibility.NOT_ELIGIBLE;
    }

    if (absoluteUniverseScopePattern.getType() != absoluteRdepsUniverseTargetPattern.getType()) {
      return Eligibility.NOT_ELIGIBLE;
    }

    switch (absoluteUniverseScopePattern.getType()) {
      case PATH_AS_TARGET:
      case SINGLE_TARGET:
        return absoluteUniverseScopePattern
                .getOriginalPattern()
                .equals(absoluteRdepsUniverseTargetPattern.getOriginalPattern())
            ? Eligibility.ELIGIBLE_AS_IS
            : Eligibility.NOT_ELIGIBLE;
      case TARGETS_IN_PACKAGE:
      case TARGETS_BELOW_DIRECTORY:
        if (!absoluteUniverseScopePattern
            .getDirectory()
            .equals(absoluteRdepsUniverseTargetPattern.getDirectory())) {
          return Eligibility.NOT_ELIGIBLE;
        }

        // Note: If we're here, both patterns are either TARGETS_IN_PACKAGE or
        // TARGETS_BELOW_DIRECTORY, and are for the same directory.

        if (absoluteUniverseScopePattern.getRulesOnly()
            == absoluteRdepsUniverseTargetPattern.getRulesOnly()) {
          return Eligibility.ELIGIBLE_AS_IS;
        }

        return absoluteUniverseScopePattern.getRulesOnly()
            // If the actual universe is narrower, then allrdeps would be unsound because it may
            // produce narrower results.
            ? Eligibility.NOT_ELIGIBLE
            // If the actual universe is wider, then allrdeps would produce wider results.
            // Therefore, we'd want to filter those results.
            : Eligibility.ELIGIBLE_WITH_FILTERING;
    }
    throw new IllegalStateException(absoluteUniverseScopePattern.getType().toString());
  }
}
