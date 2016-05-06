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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.ParseFailureListener;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Skyframe-based target pattern parsing.
 */
final class SkyframeTargetPatternEvaluator implements TargetPatternEvaluator {
  private final SkyframeExecutor skyframeExecutor;
  private String offset = "";

  SkyframeTargetPatternEvaluator(SkyframeExecutor skyframeExecutor) {
    this.skyframeExecutor = skyframeExecutor;
  }

  @Override
  public ResolvedTargets<Target> parseTargetPatternList(EventHandler eventHandler,
      List<String> targetPatterns, FilteringPolicy policy, boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    return parseTargetPatternList(offset, eventHandler, targetPatterns, policy, keepGoing);
  }

  @Override
  public ResolvedTargets<Target> parseTargetPattern(EventHandler eventHandler,
      String pattern, boolean keepGoing) throws TargetParsingException, InterruptedException {
    return parseTargetPatternList(eventHandler, ImmutableList.of(pattern),
        DEFAULT_FILTERING_POLICY, keepGoing);
  }

  @Override
  public void updateOffset(PathFragment relativeWorkingDirectory) {
    offset = relativeWorkingDirectory.getPathString();
  }

  @Override
  public String getOffset() {
    return offset;
  }

  @Override
  public Map<String, ResolvedTargets<Target>> preloadTargetPatterns(EventHandler eventHandler,
      Collection<String> patterns, boolean keepGoing)
          throws TargetParsingException, InterruptedException {
    // TODO(bazel-team): This is used only in "blaze query". There are plans to dramatically change
    // how query works on Skyframe, in which case this method is likely to go away.
    // We cannot use an ImmutableMap here because there may be null values.
    Map<String, ResolvedTargets<Target>> result = Maps.newHashMapWithExpectedSize(patterns.size());
    for (String pattern : patterns) {
      // TODO(bazel-team): This could be parallelized to improve performance. [skyframe-loading]
      result.put(pattern, parseTargetPattern(eventHandler, pattern, keepGoing));
    }
    return result;
  }

  /**
   * Loads a list of target patterns (eg, "foo/..."). When policy is set to FILTER_TESTS,
   * test_suites are going to be expanded.
   */
  ResolvedTargets<Target> parseTargetPatternList(String offset, EventHandler eventHandler,
      List<String> targetPatterns, FilteringPolicy policy, boolean keepGoing)
      throws InterruptedException, TargetParsingException {
    Iterable<TargetPatternSkyKeyOrException> keysMaybe =
        TargetPatternValue.keys(targetPatterns, policy, offset);

    ImmutableList.Builder<SkyKey> builder = ImmutableList.builder();
    for (TargetPatternSkyKeyOrException skyKeyOrException : keysMaybe) {
      try {
        builder.add(skyKeyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        if (!keepGoing) {
          throw e;
        }
        String pattern = skyKeyOrException.getOriginalPattern();
        eventHandler.handle(Event.error("Skipping '" + pattern + "': " + e.getMessage()));
        if (eventHandler instanceof ParseFailureListener) {
          ((ParseFailureListener) eventHandler).parsingError(pattern, e.getMessage());
        }
      }
    }
    ImmutableList<SkyKey> skyKeys = builder.build();
    return parseTargetPatternKeys(skyKeys, SkyframeExecutor.DEFAULT_THREAD_COUNT, keepGoing,
        eventHandler, createTargetPatternEvaluatorUtil(policy, eventHandler, keepGoing));
  }

  private TargetPatternsResultBuilder createTargetPatternEvaluatorUtil(FilteringPolicy policy,
      EventHandler eventHandler, boolean keepGoing) {
    return policy == FilteringPolicies.FILTER_TESTS
        ? new TestTargetPatternsResultBuilder(skyframeExecutor.getPackageManager(), eventHandler,
          keepGoing)
        : new BuildTargetPatternsResultBuilder();
  }

  ResolvedTargets<Target> parseTargetPatternKeys(Iterable<SkyKey> patternSkyKeys, int numThreads,
      boolean keepGoing, EventHandler eventHandler,
      TargetPatternsResultBuilder finalTargetSetEvaluator)
      throws InterruptedException, TargetParsingException {
    EvaluationResult<TargetPatternValue> result =
        skyframeExecutor.targetPatterns(patternSkyKeys, numThreads, keepGoing, eventHandler);

    String errorMessage = null;
    for (SkyKey key : patternSkyKeys) {
      TargetPatternValue resultValue = result.get(key);
      if (resultValue != null) {
        ResolvedTargets<Label> results = resultValue.getTargets();
        if (((TargetPatternValue.TargetPatternKey) key.argument()).isNegative()) {
          finalTargetSetEvaluator.addLabelsOfNegativePattern(results);
        } else {
          finalTargetSetEvaluator.addLabelsOfPositivePattern(results);
        }
      } else {
        TargetPatternValue.TargetPatternKey patternKey =
            (TargetPatternValue.TargetPatternKey) key.argument();
        String rawPattern = patternKey.getPattern();
        ErrorInfo error = result.errorMap().get(key);
        if (error == null) {
          Preconditions.checkState(!keepGoing);
          continue;
        }
        if (error.getException() != null) {
          // This exception may not be a TargetParsingException because in a nokeep_going build, the
          // target pattern parser may swallow a NoSuchPackageException but the framework will
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
        }
        finalTargetSetEvaluator.setError();

        if (eventHandler instanceof ParseFailureListener) {
          ParseFailureListener parseListener = (ParseFailureListener) eventHandler;
          parseListener.parsingError(rawPattern,  errorMessage);
        }
      }
    }

    if (result.hasError()) {
      Preconditions.checkState(errorMessage != null, "unexpected errors: %s", result.errorMap());
      finalTargetSetEvaluator.setError();
      if (!keepGoing) {
        throw new TargetParsingException(errorMessage);
      }
    }
    WalkableGraph walkableGraph = Preconditions.checkNotNull(result.getWalkableGraph(), result);
    return finalTargetSetEvaluator.build(walkableGraph);
  }
}
