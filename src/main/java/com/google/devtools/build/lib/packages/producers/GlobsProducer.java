// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.producers;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.GlobDescriptor;
import com.google.devtools.build.lib.skyframe.GlobsValue;
import com.google.devtools.build.lib.skyframe.GlobsValue.GlobRequest;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/**
 * Serves as the entrance {@link StateMachine} for {@link
 * com.google.devtools.build.lib.skyframe.GlobsFunction}.
 *
 * <p>{@link GlobsProducer} is expected to make glob computations within a package have some
 * structured logical concurrency and reduce the number of Skyframe restarts.
 */
public class GlobsProducer implements StateMachine, GlobComputationProducer.ResultSink {

  /**
   * Propagates all glob matching {@link PathFragment}s or any {@link Exception}.
   *
   * <p>See {@link GlobComputationProducer.ResultSink} for more details.
   */
  public interface ResultSink {
    void acceptAggregateMatchingPaths(ImmutableSet<PathFragment> globsMatchingResult);

    void acceptGlobError(GlobError globError);
  }

  // -------------------- Input --------------------
  private final GlobsValue.Key globsKey;
  private final ResultSink resultSink;

  // -------------------- Internal State --------------------
  private final ImmutableSet.Builder<PathFragment> aggregateMatchingPathsBuilder =
      ImmutableSet.builder();
  private final ImmutableSet<PathFragment> ignoredPackagePrefixPatterns;
  private final ConcurrentHashMap<String, Pattern> regexPatternCache;

  public GlobsProducer(
      GlobsValue.Key globsKey,
      ImmutableSet<PathFragment> ignoredPackagePrefixPatterns,
      ConcurrentHashMap<String, Pattern> regexPatternCache,
      ResultSink resultSink) {
    this.globsKey = globsKey;
    this.ignoredPackagePrefixPatterns = ignoredPackagePrefixPatterns;
    this.regexPatternCache = regexPatternCache;
    this.resultSink = resultSink;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    Preconditions.checkNotNull(ignoredPackagePrefixPatterns);
    for (GlobRequest globRequest : globsKey.getGlobRequests()) {
      GlobDescriptor globDescriptor =
          GlobDescriptor.create(
              globsKey.getPackageIdentifier(),
              globsKey.getPackageRoot(),
              PathFragment.EMPTY_FRAGMENT,
              globRequest.getPattern(),
              globRequest.getGlobOeration());
      tasks.enqueue(
          new GlobComputationProducer(
              globDescriptor,
              ignoredPackagePrefixPatterns,
              regexPatternCache,
              (GlobComputationProducer.ResultSink) this));
    }

    return this::aggregateResults;
  }

  @Override
  public void acceptPathFragmentsWithoutPackageFragment(ImmutableSet<PathFragment> pathFragments) {
    aggregateMatchingPathsBuilder.addAll(pathFragments);
  }

  @Override
  public void acceptGlobError(GlobError error) {
    resultSink.acceptGlobError(error);
  }

  public StateMachine aggregateResults(Tasks tasks) {
    resultSink.acceptAggregateMatchingPaths(aggregateMatchingPathsBuilder.build());
    return DONE;
  }
}
