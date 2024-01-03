// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.producers.GlobError;
import com.google.devtools.build.lib.packages.producers.GlobsProducer;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.Driver;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} for {@link GlobsValue}, which drives the glob matching process for all
 * globs within a package.
 *
 * <p>{@link GlobsFunction} creates a {@link GlobsProducer} which takes in all package's {@link
 * com.google.devtools.build.lib.skyframe.GlobsValue.GlobRequest}s. The {@link GlobsProducer} then
 * creates {@link com.google.devtools.build.lib.packages.producers.GlobComputationProducer} for each
 * {@link com.google.devtools.build.lib.skyframe.GlobsValue.GlobRequest} in the package and collects
 * matching paths or the first discovered error.
 *
 * <p>{@link GlobsFunction} has two benefits over {@link GlobFunction}:
 *
 * <ul>
 *   <li>The multiple GLOB nodes rdeping on the package are aggregated into a single one. This
 *       reduces some memory overhead, especially when number of globs defined in the BUILD file is
 *       very large.
 *   <li>Evaluating all globs within a package starts to have some structured logical concurrency,
 *       thus reducing the number of Skyframe restarts.
 * </ul>
 */
public class GlobsFunction implements SkyFunction {

  protected ConcurrentHashMap<String, Pattern> regexPatternCache = new ConcurrentHashMap<>();

  private static class State implements SkyKeyComputeState, GlobsProducer.ResultSink {
    @Nullable private Driver globsDriver;

    private ImmutableSet<PathFragment> globsMatchingResult;
    private GlobError error;

    @Override
    public void acceptAggregateMatchingPaths(ImmutableSet<PathFragment> globsMatchingResult) {
      this.globsMatchingResult = globsMatchingResult;
    }

    @Override
    public void acceptGlobError(GlobError error) {
      this.error = error;
    }
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    GlobsValue.Key globsKey = (GlobsValue.Key) skyKey;
    State state = env.getState(State::new);

    if (state.globsDriver == null) {
      state.globsDriver =
          new Driver(
              new GlobsProducer(globsKey, regexPatternCache, (GlobsProducer.ResultSink) state));
    }

    if (!state.globsDriver.drive(env)) {
      GlobException.handleExceptions(state.error);
      return null;
    }

    GlobException.handleExceptions(state.error);
    return new GlobsValue(state.globsMatchingResult);
  }
}
