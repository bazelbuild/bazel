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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.GlobDescriptor;
import com.google.devtools.build.lib.skyframe.GlobsValue;
import com.google.devtools.build.lib.skyframe.GlobsValue.GlobRequest;
import com.google.devtools.build.lib.skyframe.IgnoredPackagePrefixesValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * Serves as the entrance {@link StateMachine} for {@link
 * com.google.devtools.build.lib.skyframe.GlobsFunction}.
 *
 * <p>{@link GlobsValue.Key} which includes all globs within in a package is provided. {@link
 * GlobsProducer} queries the {@link #ignoredPackagePrefixPatterns} before creating {@link
 * GlobComputationProducer}s for each glob.
 *
 * <p>{@link GlobsProducer} is expected to make glob computations within a package have some
 * structured logical concurrency and reduce the number of Skyframe restarts.
 */
public class GlobsProducer
    implements StateMachine, Consumer<SkyValue>, GlobComputationProducer.ResultSink {

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
  private ImmutableSet<PathFragment> ignoredPackagePrefixPatterns = null;
  private final ConcurrentHashMap<String, Pattern> regexPatternCache;

  public GlobsProducer(
      GlobsValue.Key globsKey,
      ConcurrentHashMap<String, Pattern> regexPatternCache,
      ResultSink resultSink) {
    this.globsKey = globsKey;
    this.regexPatternCache = regexPatternCache;
    this.resultSink = resultSink;
  }

  @Override
  public StateMachine step(Tasks tasks) throws InterruptedException {
    RepositoryName repositoryName = globsKey.getPackageIdentifier().getRepository();
    tasks.lookUp(IgnoredPackagePrefixesValue.key(repositoryName), (Consumer<SkyValue>) this);
    return this::createGlobComputationProducers;
  }

  @Override
  public void accept(SkyValue skyValue) {
    this.ignoredPackagePrefixPatterns = ((IgnoredPackagePrefixesValue) skyValue).getPatterns();
  }

  public StateMachine createGlobComputationProducers(Tasks tasks) {
    if (ignoredPackagePrefixPatterns == null) {
      return DONE;
    }

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
