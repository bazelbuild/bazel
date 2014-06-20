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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.pkgcache.ParseFailureListener;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.UpdateResult;

import java.util.List;

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
  public ResolvedTargets<Target> parseTargetPatternList(ErrorEventListener listener,
      List<String> targetPatterns, FilteringPolicy policy, boolean keepGoing)
      throws TargetParsingException, InterruptedException {
    return parseTargetPatternList(offset, listener, targetPatterns, policy, keepGoing);
  }

  @Override
  public ResolvedTargets<Target> parseTargetPattern(ErrorEventListener listener,
      String pattern, boolean keepGoing) throws TargetParsingException, InterruptedException {
    return parseTargetPatternList(listener, ImmutableList.of(pattern),
        FilteringPolicies.NO_FILTER, keepGoing);
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
  public List<ResolvedTargets<Target>> preloadTargetPatterns(ErrorEventListener listener,
      List<String> patterns, boolean keepGoing)
          throws TargetParsingException, InterruptedException {
    // TODO(bazel-team): This is used only in "blaze query". There are plans to dramatically change
    // how query works on Skyframe, in which case this method is likely to go away.
    ImmutableList.Builder<ResolvedTargets<Target>> result = ImmutableList.builder();
    for (String pattern : patterns) {
      // TODO(bazel-team): This could be parallelized to improve performance. [skyframe-loading]
      result.add(parseTargetPattern(listener, pattern, keepGoing));
    }
    return result.build();
  }

  /**
   * Loads a list of target patterns (eg, "foo/...").
   */
  ResolvedTargets<Target> parseTargetPatternList(String offset, ErrorEventListener listener,
      List<String> targetPatterns, FilteringPolicy policy, boolean keepGoing)
      throws InterruptedException, TargetParsingException {
    Iterable<NodeKey> patternNodeKeys = TargetPatternNode.keys(targetPatterns, policy, offset);
    UpdateResult<TargetPatternNode> result =
        skyframeExecutor.targetPatterns(patternNodeKeys, keepGoing, listener);

    String errorMessage = null;
    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    for (NodeKey key : patternNodeKeys) {
      TargetPatternNode resultNode = result.get(key);
      if (resultNode != null) {
        ResolvedTargets<Target> results = resultNode.getTargets();
        if (((TargetPatternNode.TargetPattern) key.getNodeName()).isNegative()) {
          builder.filter(Predicates.not(Predicates.in(results.getTargets())));
        } else {
          builder.merge(results);
        }
      } else {
        TargetPatternNode.TargetPattern pattern =
            (TargetPatternNode.TargetPattern) key.getNodeName();
        String rawPattern = pattern.getPattern();
        ErrorInfo error = result.errorMap().get(key);
        if (error == null) {
          Preconditions.checkState(!keepGoing);
          continue;
        }
        if (error.getException() != null) {
          errorMessage = error.getException().getMessage();
        } else if (!Iterables.isEmpty(error.getCycleInfo())) {
          errorMessage = "cycles detected during target parsing";
          skyframeExecutor.getCyclesReporter().reportCycles(error.getCycleInfo(), key, listener);
        } else {
          throw new IllegalStateException(error.toString());
        }
        if (keepGoing) {
          listener.error(null, "Skipping '" + rawPattern + "': " + errorMessage);
        }
        builder.setError();

        if (listener instanceof ParseFailureListener) {
          ParseFailureListener parseListener = (ParseFailureListener) listener;
          parseListener.parsingError(rawPattern,  errorMessage);
        }
      }
    }

    if (!keepGoing && result.hasError()) {
      Preconditions.checkState(errorMessage != null, "unexpected errors: %s", result.errorMap());
      throw new TargetParsingException(errorMessage);
    }
    return builder.build();
  }
}
