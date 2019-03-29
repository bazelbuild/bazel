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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.ParsingFailedEvent;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.pkgcache.TargetPatternPreloader;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Skyframe-based target pattern parsing.
 */
final class SkyframeTargetPatternEvaluator
    implements TargetPatternEvaluator, TargetPatternPreloader {
  private final SkyframeExecutor skyframeExecutor;

  SkyframeTargetPatternEvaluator(SkyframeExecutor skyframeExecutor) {
    this.skyframeExecutor = skyframeExecutor;
  }

  @Override
  public ResolvedTargets<Target> parseTargetPatternList(
      PathFragment relativeWorkingDirectory,
      ExtendedEventHandler eventHandler,
      List<String> targetPatterns,
      FilteringPolicy policy,
      boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    return parseTargetPatternList(
        relativeWorkingDirectory.getPathString(),
        eventHandler,
        ImmutableList.copyOf(targetPatterns),
        policy,
        keepGoing);
  }

  @Override
  public Map<String, ResolvedTargets<Target>> preloadTargetPatterns(
      ExtendedEventHandler eventHandler,
      PathFragment relativeWorkingDirectory,
      Collection<String> patterns,
      boolean keepGoing)
          throws TargetParsingException, InterruptedException {
    // TODO(bazel-team): This is used only in "blaze query". There are plans to dramatically change
    // how query works on Skyframe, in which case this method is likely to go away.
    ImmutableList.Builder<TargetPatternsAndKeysAndResultBuilder>
        targetPatternsAndKeysAndResultListBuilder = ImmutableList.builder();
    FilteringPolicy policy = DEFAULT_FILTERING_POLICY;
    for (String pattern : patterns) {
      ImmutableList<String> singletonPatternList = ImmutableList.of(pattern);
      targetPatternsAndKeysAndResultListBuilder.add(new TargetPatternsAndKeysAndResultBuilder(
          singletonPatternList,
          getTargetPatternKeys(
              relativeWorkingDirectory.getPathString(),
              eventHandler,
              singletonPatternList,
              policy,
              keepGoing),
          createTargetPatternEvaluatorUtil(policy, eventHandler, keepGoing)));

    }
    ImmutableList<ResolvedTargets<Target>> batchResult = parseTargetPatternKeysBatch(
        targetPatternsAndKeysAndResultListBuilder.build(),
        SkyframeExecutor.DEFAULT_THREAD_COUNT,
        keepGoing,
        eventHandler);
    Preconditions.checkState(patterns.size() == batchResult.size(), patterns);
    ImmutableMap.Builder<String, ResolvedTargets<Target>> resultBuilder = ImmutableMap.builder();
    int i = 0;
    for (String pattern : patterns) {
      resultBuilder.put(pattern, batchResult.get(i++));
    }
    return resultBuilder.build();
  }

  private Iterable<TargetPatternKey> getTargetPatternKeys(
      String offset,
      ExtendedEventHandler eventHandler,
      ImmutableList<String> targetPatterns,
      FilteringPolicy policy,
      boolean keepGoing) throws TargetParsingException {
    // TODO: it's possible all of these are ok but really, figure it out
    ImmutableMap<RepositoryName, ImmutableList<String>> patternsMap = ImmutableMap.of(RepositoryName.MAIN, ImmutableList.copyOf(targetPatterns));
    Iterable<TargetPatternSkyKeyOrException> keysMaybe =
        TargetPatternValue.keys(patternsMap, policy, offset);
    ImmutableList.Builder<TargetPatternKey> builder = ImmutableList.builder();
    for (TargetPatternSkyKeyOrException skyKeyOrException : keysMaybe) {
      try {
        builder.add(skyKeyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        // We report a parsing failed exception to the event bus here in case the pattern did not
        // successfully parse (which happens before the SkyKey is created). Otherwise the
        // TargetPatternFunction posts the event.
        eventHandler.post(
            new ParsingFailedEvent(skyKeyOrException.getOriginalPattern(),  e.getMessage()));
        if (!keepGoing) {
          throw e;
        }
        String pattern = skyKeyOrException.getOriginalPattern();
        eventHandler.handle(Event.error("Skipping '" + pattern + "': " + e.getMessage()));
      }
    }
    return builder.build();
  }

  /**
   * Loads a list of target patterns (eg, "foo/..."). When policy is set to FILTER_TESTS,
   * test_suites are going to be expanded.
   */
  private ResolvedTargets<Target> parseTargetPatternList(
      String offset,
      ExtendedEventHandler eventHandler,
      ImmutableList<String> targetPatterns,
      FilteringPolicy policy,
      boolean keepGoing)
      throws InterruptedException, TargetParsingException {
    return Iterables.getOnlyElement(
        parseTargetPatternKeysBatch(
            ImmutableList.of(
                new TargetPatternsAndKeysAndResultBuilder(
                    targetPatterns,
                    getTargetPatternKeys(offset, eventHandler, targetPatterns, policy, keepGoing),
                    createTargetPatternEvaluatorUtil(policy, eventHandler, keepGoing))),
            SkyframeExecutor.DEFAULT_THREAD_COUNT,
            keepGoing,
            eventHandler));
  }

  private TargetPatternsResultBuilder createTargetPatternEvaluatorUtil(
      FilteringPolicy policy, ExtendedEventHandler eventHandler, boolean keepGoing) {
    return policy == FilteringPolicies.FILTER_TESTS
        ? new TestTargetPatternsResultBuilder(
            skyframeExecutor.getPackageManager(), eventHandler, keepGoing)
        : new BuildTargetPatternsResultBuilder();
  }

  private class TargetPatternsAndKeysAndResultBuilder {
    private final ImmutableList<String> targetPatterns;
    private final Iterable<TargetPatternKey> patternSkyKeys;
    private final TargetPatternsResultBuilder resultBuilder;

    private TargetPatternsAndKeysAndResultBuilder(
        ImmutableList<String> targetPatterns,
        Iterable<TargetPatternKey> patternSkyKeys,
        TargetPatternsResultBuilder resultBuilder) {
      this.targetPatterns = targetPatterns;
      this.patternSkyKeys = patternSkyKeys;
      this.resultBuilder = resultBuilder;
    }
  }

  private ImmutableList<ResolvedTargets<Target>> parseTargetPatternKeysBatch(
      ImmutableList<TargetPatternsAndKeysAndResultBuilder> targetPatternsAndKeysAndResultBuilders,
      int numThreads,
      boolean keepGoing,
      ExtendedEventHandler eventHandler)
      throws InterruptedException, TargetParsingException {
    ImmutableList.Builder<TargetPatternKey> allKeysBuilder = ImmutableList.builder();
    for (TargetPatternsAndKeysAndResultBuilder targetPatternsAndKeysAndResultBuilder
        : targetPatternsAndKeysAndResultBuilders) {
      allKeysBuilder.addAll(targetPatternsAndKeysAndResultBuilder.patternSkyKeys);
    }
    EvaluationResult<TargetPatternValue> result = skyframeExecutor.targetPatterns(
        allKeysBuilder.build(), numThreads, keepGoing, eventHandler);
    WalkableGraph walkableGraph = Preconditions.checkNotNull(result.getWalkableGraph(), result);
    ImmutableList.Builder<ResolvedTargets<Target>> resolvedTargetsListBuilder =
        ImmutableList.builder();
    for (TargetPatternsAndKeysAndResultBuilder targetPatternsAndKeysAndResultBuilder
        : targetPatternsAndKeysAndResultBuilders) {
      ImmutableList<String> targetPatterns = targetPatternsAndKeysAndResultBuilder.targetPatterns;
      Iterable<TargetPatternKey> patternSkyKeys =
          targetPatternsAndKeysAndResultBuilder.patternSkyKeys;
      TargetPatternsResultBuilder resultBuilder =
          targetPatternsAndKeysAndResultBuilder.resultBuilder;
      String errorMessage = null;
      boolean hasError = false;
      for (TargetPatternKey key : patternSkyKeys) {
        TargetPatternValue resultValue = result.get(key);
        if (resultValue != null) {
          ResolvedTargets<Label> results = resultValue.getTargets();
          if (key.isNegative()) {
            resultBuilder.addLabelsOfNegativePattern(results);
          } else {
            resultBuilder.addLabelsOfPositivePattern(results);
          }
        } else {
          String rawPattern = key.getPattern();
          ErrorInfo error = result.errorMap().get(key);
          if (error == null) {
            Preconditions.checkState(!keepGoing);
            continue;
          }
          hasError = true;
          if (error.getException() != null) {
            // This exception may not be a TargetParsingException because in a nokeep_going build,
            // the target pattern parser may swallow a NoSuchPackageException but the framework will
            // bubble it up anyway.
            Preconditions.checkArgument(!keepGoing
                || error.getException() instanceof TargetParsingException, error);
            errorMessage = error.getException().getMessage();
          } else if (!Iterables.isEmpty(error.getCycleInfo())) {
            errorMessage = "cycles detected during target parsing";
            skyframeExecutor.getCyclesReporter().reportCycles(
                error.getCycleInfo(), key, eventHandler);
          } else {
            throw new IllegalStateException(error.toString());
          }
          if (keepGoing) {
            eventHandler.handle(Event.error("Skipping '" + rawPattern + "': " + errorMessage));
            eventHandler.post(PatternExpandingError.skipped(rawPattern, errorMessage));
          }
          resultBuilder.setError();
        }
      }

      if (hasError) {
        Preconditions.checkState(errorMessage != null, "unexpected errors: %s", result.errorMap());
        resultBuilder.setError();
        if (!keepGoing) {
          eventHandler.post(PatternExpandingError.failed(targetPatterns, errorMessage));
          throw new TargetParsingException(errorMessage);
        }
      }
      resolvedTargetsListBuilder.add(resultBuilder.build(walkableGraph));
    }
    return resolvedTargetsListBuilder.build();
  }
}
