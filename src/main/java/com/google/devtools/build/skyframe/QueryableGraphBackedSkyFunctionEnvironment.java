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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A {@link SkyFunction.Environment} backed by a {@link QueryableGraph}. For use when a single
 * SkyFunction needs recomputation, and its dependencies do not need to be evaluated. Any missing
 * dependencies will be ignored.
 */
public class QueryableGraphBackedSkyFunctionEnvironment extends AbstractSkyFunctionEnvironment {
  private final QueryableGraph queryableGraph;
  private final EventHandler eventHandler;

  public QueryableGraphBackedSkyFunctionEnvironment(
      QueryableGraph queryableGraph, EventHandler eventHandler) {
    this.queryableGraph = queryableGraph;
    this.eventHandler = eventHandler;
  }

  private static final Function<NodeEntry, ValueOrUntypedException> NODE_ENTRY_TO_UNTYPED_VALUE =
      new Function<NodeEntry, ValueOrUntypedException>() {
        @Override
        public ValueOrUntypedException apply(@Nullable NodeEntry nodeEntry) {
          if (nodeEntry == null || !nodeEntry.isDone()) {
            return ValueOrExceptionUtils.ofNull();
          }
          SkyValue maybeWrappedValue = nodeEntry.getValueMaybeWithMetadata();
          SkyValue justValue = ValueWithMetadata.justValue(maybeWrappedValue);
          if (justValue != null) {
            return ValueOrExceptionUtils.ofValueUntyped(justValue);
          }
          ErrorInfo errorInfo =
              Preconditions.checkNotNull(ValueWithMetadata.getMaybeErrorInfo(maybeWrappedValue));
          Exception exception = errorInfo.getException();

          if (exception != null) {
            // Give SkyFunction#compute a chance to handle this exception.
            return ValueOrExceptionUtils.ofExn(exception);
          }
          // In a cycle.
          Preconditions.checkState(
              !Iterables.isEmpty(errorInfo.getCycleInfo()), "%s %s", errorInfo, maybeWrappedValue);
          return ValueOrExceptionUtils.ofNull();
        }
      };

  @Override
  protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(Set<SkyKey> depKeys) {
    Map<SkyKey, NodeEntry> resultMap = queryableGraph.getBatch(depKeys);
    Map<SkyKey, NodeEntry> resultWithMissingKeys = new HashMap<>(resultMap);
    for (SkyKey missingDep : Sets.difference(depKeys, resultMap.keySet())) {
      resultWithMissingKeys.put(missingDep, null);
    }
    return Maps.transformValues(resultWithMissingKeys, NODE_ENTRY_TO_UNTYPED_VALUE);
  }

  @Override
  public EventHandler getListener() {
    return eventHandler;
  }

  @Override
  public boolean inErrorBubblingForTesting() {
    return false;
  }
}
