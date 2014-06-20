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
package com.google.devtools.build.skyframe;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * The result of a Skyframe {@link Evaluator#eval} call. Will contain all the
 * successfully evaluated nodes, retrievable through {@link #get}. As well, the {@link ErrorInfo}
 * for the first node that failed to evaluate (in the non-keep-going case), or any remaining nodes
 * that failed to evaluate (in the keep-going case) will be retrievable.
 *
 * @param <T> The type of the nodes that the caller has requested.
 */
public class UpdateResult<T extends Node> {

  private final boolean hasError;

  private final Map<NodeKey, T> resultMap;
  private final Map<NodeKey, ErrorInfo> errorMap;

  /**
   * Constructor for the "completed" case. Used only by {@link Builder}.
   */
  private UpdateResult(Map<NodeKey, T> result, Map<NodeKey, ErrorInfo> errorMap, boolean hasError) {
    Preconditions.checkState(errorMap.isEmpty() || hasError,
        "result=%s, errorMap=%s", result, errorMap);
    this.resultMap = Preconditions.checkNotNull(result);
    this.errorMap = Preconditions.checkNotNull(errorMap);
    this.hasError = hasError;
  }

  /**
   * Get a successfully evaluated node.
   */
  public T get(NodeKey key) {
    Preconditions.checkNotNull(resultMap, key);
    return resultMap.get(key);
  }

  /**
   * @return Whether or not the eval successfully evaluated all requested nodes. Note that this
   * may return true even if all nodes returned are available in get(). This happens if a top-level
   * node depends transitively on some node that recovered from a {@link NodeBuilderException}.
   */
  public boolean hasError() {
    return hasError;
  }

  /**
   * @return All successfully evaluated {@link Node}s.
   */
  public Collection<T> values() {
    return Collections.unmodifiableCollection(resultMap.values());
  }

  /**
   * Returns {@link Map} of {@link NodeKey}s to {@link ErrorInfo}. Note that currently some
   * of the returned NodeKeys may not be the ones requested by the user. Moreover, the NodeKey
   * is not necessarily the cause of the error -- it is just the node that was being evaluated
   * when the error was discovered. For the cause of the error, use
   * {@link ErrorInfo#getRootCauses()} on each ErrorInfo.
   */
  public Map<NodeKey, ErrorInfo> errorMap() {
    return ImmutableMap.copyOf(errorMap);
  }

  /**
   * @param key {@link NodeKey} to get {@link ErrorInfo} for.
   */
  public ErrorInfo getError(NodeKey key) {
    return Preconditions.checkNotNull(errorMap, key).get(key);
  }

  /**
   * @return Names of all nodes that were successfully evaluated.
   */
  public <S> Collection<? extends S> keyNames() {
    return this.<S>getNames(resultMap.keySet());
  }

  @SuppressWarnings("unchecked")
  private <S> Collection<? extends S> getNames(Collection<NodeKey> keys) {
    Collection<S> names = Lists.newArrayListWithCapacity(keys.size());
    for (NodeKey key : keys) {
      names.add((S) key.getNodeName());
    }
    return names;
  }

  /**
   * Returns some error info. Convenience method equivalent to
   * Iterables.getFirst({@link #errorMap()}, null).getValue().
   */
  public ErrorInfo getError() {
    return Iterables.getFirst(errorMap.entrySet(), null).getValue();
  }

  @Override
  @SuppressWarnings("deprecation")
  public String toString() {
    return Objects.toStringHelper(this)  // MoreObjects is not in Guava
        .add("hasError", hasError)
        .add("errorMap", errorMap)
        .add("resultMap", resultMap)
        .toString();
  }

  static <T extends Node> Builder<T> builder() {
    return new Builder<>();
  }

  static class Builder<T extends Node> {
    private final Map<NodeKey, T> result = new HashMap<>();
    private final Map<NodeKey, ErrorInfo> errors = new HashMap<>();
    private boolean hasError = false;

    @SuppressWarnings("unchecked")
    Builder<T> addResult(NodeKey key, Node node) {
      result.put(key, Preconditions.checkNotNull((T) node, key));
      return this;
    }

    Builder<T> addError(NodeKey key, ErrorInfo error) {
      errors.put(key, Preconditions.checkNotNull(error, key));
      return this;
    }

    UpdateResult<T> build() {
      return new UpdateResult<>(result, errors, hasError);
    }

    public void setHasError(boolean hasError) {
      this.hasError = hasError;
    }
  }
}
