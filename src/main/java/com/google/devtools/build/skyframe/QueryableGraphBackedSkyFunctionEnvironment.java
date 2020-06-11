// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.Map;

/**
 * A {@link SkyFunction.Environment} backed by a {@link QueryableGraph}. For use when a single
 * SkyFunction needs recomputation, and its dependencies do not need to be evaluated. Any missing
 * dependencies will be ignored.
 */
public class QueryableGraphBackedSkyFunctionEnvironment extends AbstractSkyFunctionEnvironment {
  private final QueryableGraph queryableGraph;
  private final ExtendedEventHandler eventHandler;

  public QueryableGraphBackedSkyFunctionEnvironment(
      QueryableGraph queryableGraph, ExtendedEventHandler eventHandler) {
    this.queryableGraph = queryableGraph;
    this.eventHandler = eventHandler;
  }

  private ValueOrUntypedException toUntypedValue(NodeEntry nodeEntry) throws InterruptedException {
    if (nodeEntry == null || !nodeEntry.isDone()) {
      valuesMissing = true;
      return ValueOrUntypedException.ofNull();
    }
    SkyValue maybeWrappedValue = nodeEntry.getValueMaybeWithMetadata();
    SkyValue justValue = ValueWithMetadata.justValue(maybeWrappedValue);
    if (justValue != null) {
      return ValueOrUntypedException.ofValueUntyped(justValue);
    }
    errorMightHaveBeenFound = true;
    ErrorInfo errorInfo =
        Preconditions.checkNotNull(ValueWithMetadata.getMaybeErrorInfo(maybeWrappedValue));
    Exception exception = errorInfo.getException();

    if (exception != null) {
      // Give SkyFunction#compute a chance to handle this exception.
      return ValueOrUntypedException.ofExn(exception);
    }
    // In a cycle.
    Preconditions.checkState(
        !errorInfo.getCycleInfo().isEmpty(), "%s %s", errorInfo, maybeWrappedValue);
    return ValueOrUntypedException.ofNull();
  }

  @Override
  protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> resultMap =
        queryableGraph.getBatch(null, Reason.DEP_REQUESTED, depKeys);
    // resultMap will be smaller than what we actually return if some of depKeys were not found in
    // the graph. Pad to a minimum of 16 to avoid excessive resizing.
    Map<SkyKey, ValueOrUntypedException> result =
        Maps.newHashMapWithExpectedSize(Math.max(16, resultMap.size()));
    for (SkyKey dep : depKeys) {
      result.put(dep, toUntypedValue(resultMap.get(dep)));
    }
    return result;
  }

  @Override
  public ExtendedEventHandler getListener() {
    return eventHandler;
  }

  @Override
  public boolean inErrorBubblingForTesting() {
    return false;
  }

  @Override
  public boolean restartPermitted() {
    return false;
  }
}
