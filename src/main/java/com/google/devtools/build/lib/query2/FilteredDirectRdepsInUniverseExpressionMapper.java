// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.query2.engine.BinaryOperatorExpression;
import com.google.devtools.build.lib.query2.engine.FunctionExpression;
import com.google.devtools.build.lib.query2.engine.LetExpression;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.FilteringQueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionMapper;
import com.google.devtools.build.lib.query2.engine.QueryExpressionVisitor;
import com.google.devtools.build.lib.query2.engine.RdepsFunction;
import com.google.devtools.build.lib.query2.engine.SetExpression;
import com.google.devtools.build.lib.query2.engine.TargetLiteral;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link QueryExpressionMapper} that transforms each occurrence of an expression of the form
 * {@literal 'rdeps(filterfunc(<pat>, <universeScope>), <T>, 1)'} to {@literal 'filterfunc(<pat>,
 * rdeps(<universeScope>, <T>, 1))'}.
 *
 * <p>By factoring the filterfunc out of the rdeps universe, we prepare the query for transformation
 * by the {@link RdepsToAllRdepsQueryExpressionMapper}.
 *
 * <p><em>Note:</em> we require a max depth of 1 because the transformation is only sound with a
 * depth of 1. Consider a query of depth 2.
 *
 * <ol>
 *   <li>{@literal 'rdeps(kind(^foo$, //...), //base:b, 2)'} yields these two disjoint sets:
 *       <ol>
 *         <li>targets of kind 'foo' that depend directly on //base:b
 *         <li>targets of kind 'foo' that do not depend directly on //base:b, and depend on a target
 *             of kind 'foo' that itself depends directly on //base:b
 *       </ol>
 *   <li>{@literal 'kind(^foo$, rdeps(//..., //base:b, 2))'} yields these two disjoint sets:
 *       <ol>
 *         <li>targets of kind 'foo' that depend directly on //base:b
 *         <li>targets of kind 'foo' that do not depend directly on //base:b, and depend on a target
 *             of <em>any</em> kind that itself depends directly on //base:b
 *       </ol>
 * </ol>
 *
 * With a depth of 1, the rdeps operator in both queries would return only the first set: proof the
 * transformation is sound. With a depth of 2, we see the rdeps operator with a filtered universe
 * returns a strictly smaller set.
 */
public class FilteredDirectRdepsInUniverseExpressionMapper extends QueryExpressionMapper<Void> {
  private final TargetPattern.Parser targetPatternParser;
  private final String absoluteUniverseScopePattern;

  @VisibleForTesting
  public FilteredDirectRdepsInUniverseExpressionMapper(
      TargetPattern.Parser targetPatternParser, String universeScopePattern) {
    this.targetPatternParser = targetPatternParser;
    this.absoluteUniverseScopePattern = targetPatternParser.absolutize(universeScopePattern);
  }

  @Override
  public QueryExpression visit(FunctionExpression functionExpression, Void context) {
    if (functionExpression.getFunction().getName().equals(new RdepsFunction().getName())) {
      List<Argument> args = functionExpression.getArgs();
      // This transformation only applies to the 3-arg form of rdeps, with depth == 1.
      if (args.size() == 3 && args.get(2).getInteger() == 1) {
        List<FunctionExpression> universeFilteringFunctions =
            args.get(0).getExpression().accept(new ExtractFilteringFunctionsFromUniverseVisitor());
        // If we get back a non-empty list of FunctionExpressions, these are filtering functions
        // that can be safely factored out.
        if (!universeFilteringFunctions.isEmpty()) {
          QueryExpression curFunction = makeUnfilteredRdepsWithDepthOne(args.get(1));
          for (FunctionExpression filterExpr : universeFilteringFunctions) {
            curFunction = wrapExprWithFilter(curFunction, filterExpr);
          }
          return curFunction;
        }
      }
    }
    return super.visit(functionExpression, context);
  }

  private FunctionExpression makeUnfilteredRdepsWithDepthOne(Argument target) {
    return new FunctionExpression(
        new RdepsFunction(),
        ImmutableList.of(
            Argument.of(new TargetLiteral(absoluteUniverseScopePattern)), target, Argument.of(1)));
  }

  private static QueryExpression wrapExprWithFilter(
      QueryExpression curFunction, FunctionExpression filterExpr) {
    List<Argument> filterExprArgs = filterExpr.getArgs();
    FilteringQueryFunction filteringFunction = filterExpr.getFunction().asFilteringFunction();
    List<Argument> rewrittenArgs = new ArrayList<>();
    for (int i = 0; i < filterExprArgs.size(); i++) {
      if (i == filteringFunction.getExpressionToFilterIndex()) {
        rewrittenArgs.add(Argument.of(curFunction));
      } else {
        rewrittenArgs.add(filterExprArgs.get(i));
      }
    }
    return new FunctionExpression(filteringFunction, rewrittenArgs);
  }

  /**
   * Internal visitor applied to the universe argument of all QueryExpressions of the form {@literal
   * rdeps(u, x, 1)}.
   */
  private class ExtractFilteringFunctionsFromUniverseVisitor
      implements QueryExpressionVisitor<List<FunctionExpression>, Void> {

    @Override
    public List<FunctionExpression> visit(TargetLiteral targetLiteral, Void context) {
      return ImmutableList.of();
    }

    @Override
    public List<FunctionExpression> visit(
        BinaryOperatorExpression binaryOperatorExpression, Void context) {
      return ImmutableList.of();
    }

    @SuppressWarnings("MixedMutabilityReturnType")
    @Override
    public List<FunctionExpression> visit(FunctionExpression functionExpression, Void context) {
      FilteringQueryFunction filteringFunction =
          functionExpression.getFunction().asFilteringFunction();
      if (filteringFunction == null) {
        return ImmutableList.of();
      }
      Argument filteredArgument =
          functionExpression.getArgs().get(filteringFunction.getExpressionToFilterIndex());
      Preconditions.checkArgument(filteredArgument.getType() == ArgumentType.EXPRESSION);
      QueryExpression filteredExpression = filteredArgument.getExpression();

      ArrayList<FunctionExpression> results = new ArrayList<>();
      if (filteredExpression instanceof TargetLiteral) {
        TargetLiteral literalUniverseExpression = (TargetLiteral) filteredExpression;
        String absolutizedUniverseExpression =
            targetPatternParser.absolutize(literalUniverseExpression.getPattern());
        if (absolutizedUniverseExpression.equals(absoluteUniverseScopePattern)) {
          results.add(functionExpression);
        }
      } else if (filteredExpression instanceof FunctionExpression) {
        List<FunctionExpression> nestedFilters = filteredExpression.accept(this);
        if (!nestedFilters.isEmpty()) {
          results.addAll(nestedFilters);
          results.add(functionExpression);
        }
      }
      return results;
    }

    @Override
    public List<FunctionExpression> visit(LetExpression letExpression, Void context) {
      return ImmutableList.of();
    }

    @Override
    public List<FunctionExpression> visit(SetExpression setExpression, Void context) {
      return ImmutableList.of();
    }
  }
}
