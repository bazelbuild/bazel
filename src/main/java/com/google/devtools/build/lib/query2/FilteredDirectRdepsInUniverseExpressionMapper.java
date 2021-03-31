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
import com.google.devtools.build.lib.query2.RdepsToAllRdepsQueryExpressionMapper.Eligibility;
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
import javax.annotation.Nullable;

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
  private final TargetPattern absoluteUniverseScopePattern;

  @VisibleForTesting
  public FilteredDirectRdepsInUniverseExpressionMapper(
      TargetPattern.Parser targetPatternParser, TargetPattern absoluteUniverseScopePattern) {
    this.targetPatternParser = targetPatternParser;
    this.absoluteUniverseScopePattern = absoluteUniverseScopePattern;
  }

  @Override
  public QueryExpression visit(FunctionExpression functionExpression, Void context) {
    if (functionExpression.getFunction().getName().equals(new RdepsFunction().getName())) {
      List<Argument> args = functionExpression.getArgs();
      // This transformation only applies to the 3-arg form of rdeps, with depth == 1.
      if (args.size() == 3 && args.get(2).getInteger() == 1) {
        Argument rdepsUniverseArgument = args.get(0);
        QueryExpression rdepsUniverseExpression = rdepsUniverseArgument.getExpression();
        ExtractFilteringFunctionsResult result =
            rdepsUniverseExpression.accept(new ExtractFilteringFunctionsFromUniverseVisitor());
        // If we get back a non-empty result, then there are filtering functions that can be safely
        // factored out.
        if (result.universeArgument != null) {
          QueryExpression curFunction =
              makeUnfilteredRdepsWithDepthOne(
                  /*rdepsUniverseArgument=*/ result.universeArgument,
                  /*sourceArgument=*/ args.get(1));
          for (FunctionExpression filterExpr : result.filteringExpressions) {
            curFunction = wrapExprWithFilter(curFunction, filterExpr);
          }
          return curFunction;
        }
      }
    }
    return super.visit(functionExpression, context);
  }

  private FunctionExpression makeUnfilteredRdepsWithDepthOne(
      Argument rdepsUniverseArgument, Argument sourceArgument) {
    return new FunctionExpression(
        new RdepsFunction(),
        ImmutableList.of(rdepsUniverseArgument, sourceArgument, Argument.of(1)));
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

  private static class ExtractFilteringFunctionsResult {
    private static final ExtractFilteringFunctionsResult EMPTY =
        new ExtractFilteringFunctionsResult(ImmutableList.of(), null);

    private final ImmutableList<FunctionExpression> filteringExpressions;
    @Nullable private final Argument universeArgument;

    private ExtractFilteringFunctionsResult(
        ImmutableList<FunctionExpression> filteringExpressions,
        @Nullable Argument universeArgument) {
      this.filteringExpressions = filteringExpressions;
      this.universeArgument = universeArgument;
    }
  }

  /**
   * Internal visitor applied to the universe argument of all QueryExpressions of the form {@literal
   * rdeps(u, x, 1)}.
   */
  private class ExtractFilteringFunctionsFromUniverseVisitor
      implements QueryExpressionVisitor<ExtractFilteringFunctionsResult, Void> {

    @Override
    public ExtractFilteringFunctionsResult visit(TargetLiteral targetLiteral, Void context) {
      return ExtractFilteringFunctionsResult.EMPTY;
    }

    @Override
    public ExtractFilteringFunctionsResult visit(
        BinaryOperatorExpression binaryOperatorExpression, Void context) {
      return ExtractFilteringFunctionsResult.EMPTY;
    }

    @SuppressWarnings("MixedMutabilityReturnType")
    @Override
    public ExtractFilteringFunctionsResult visit(
        FunctionExpression functionExpression, Void context) {
      FilteringQueryFunction filteringFunction =
          functionExpression.getFunction().asFilteringFunction();
      if (filteringFunction == null) {
        return ExtractFilteringFunctionsResult.EMPTY;
      }
      Argument filteredArgument =
          functionExpression.getArgs().get(filteringFunction.getExpressionToFilterIndex());
      Preconditions.checkArgument(filteredArgument.getType() == ArgumentType.EXPRESSION);
      QueryExpression filteredExpression = filteredArgument.getExpression();

      if (filteredExpression instanceof TargetLiteral) {
        Eligibility eligibility =
            RdepsToAllRdepsQueryExpressionMapper.determineEligibility(
                targetPatternParser,
                absoluteUniverseScopePattern,
                ((TargetLiteral) filteredExpression).getPattern());
        if (eligibility == Eligibility.ELIGIBLE_AS_IS
            || eligibility == Eligibility.ELIGIBLE_WITH_FILTERING) {
          return new ExtractFilteringFunctionsResult(
              ImmutableList.of(functionExpression), filteredArgument);
        }
      } else if (filteredExpression instanceof FunctionExpression) {
        ExtractFilteringFunctionsResult recursiveResult = filteredExpression.accept(this);
        if (recursiveResult.universeArgument != null) {
          return new ExtractFilteringFunctionsResult(
              /*filteringExpressions=*/ ImmutableList.<FunctionExpression>builder()
                  .addAll(recursiveResult.filteringExpressions)
                  .add(functionExpression)
                  .build(),
              recursiveResult.universeArgument);
        }
      }
      return ExtractFilteringFunctionsResult.EMPTY;
    }

    @Override
    public ExtractFilteringFunctionsResult visit(LetExpression letExpression, Void context) {
      return ExtractFilteringFunctionsResult.EMPTY;
    }

    @Override
    public ExtractFilteringFunctionsResult visit(SetExpression setExpression, Void context) {
      return ExtractFilteringFunctionsResult.EMPTY;
    }
  }
}
