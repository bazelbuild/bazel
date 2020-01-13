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
package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.util.RegexFilter;
import java.util.ArrayList;
import java.util.List;

/**
 * Builder class to create an {@link Executor} instance. This class is part of the module API,
 * which allows modules to affect how the executor is initialized.
 */
public class ExecutorBuilder {
  private final List<ActionContextProvider> actionContextProviders = new ArrayList<>();
  private final SpawnActionContextMaps.Builder spawnActionContextMapsBuilder =
      new SpawnActionContextMaps.Builder();
  private ActionInputPrefetcher prefetcher;

  // These methods shouldn't be public, but they have to be right now as ExecutionTool is in another
  // package.
  public ImmutableList<ActionContextProvider> getActionContextProviders() {
    return ImmutableList.copyOf(actionContextProviders);
  }

  public SpawnActionContextMaps.Builder getSpawnActionContextMapsBuilder() {
    return spawnActionContextMapsBuilder;
  }

  public ActionInputPrefetcher getActionInputPrefetcher() {
    return prefetcher == null ? ActionInputPrefetcher.NONE : prefetcher;
  }

  /**
   * Adds the specified action context providers to the executor.
   */
  public ExecutorBuilder addActionContextProvider(ActionContextProvider provider) {
    this.actionContextProviders.add(provider);
    return this;
  }

  /**
   * Adds the specified action context to the executor, by wrapping it in a simple action context
   * provider implementation.
   */
  public ExecutorBuilder addActionContext(ActionContext context) {
    return addActionContextProvider(new SimpleActionContextProvider(context));
  }

  /**
   * Sets the strategy names for a given action mnemonic.
   *
   * <p>During execution, the {@link ProxySpawnActionContext} will ask each strategy whether it can
   * execute a given Spawn. The first strategy in the list that says so will get the job.
   */
  public ExecutorBuilder addStrategyByMnemonic(String mnemonic, List<String> strategies) {
    spawnActionContextMapsBuilder.strategyByMnemonicMap().replaceValues(mnemonic, strategies);
    return this;
  }

  /**
   * Sets the strategy names to use in the remote branch of dynamic execution for a given action
   * mnemonic.
   *
   * <p>During execution, each strategy is {@linkplain SpawnActionContext#canExec(Spawn,
   * com.google.devtools.build.lib.actions.ActionExecutionContext) asked} whether it can execute a
   * given Spawn. The first strategy in the list that says so will get the job.
   */
  public ExecutorBuilder addDynamicRemoteStrategiesByMnemonic(
      String mnemonic, List<String> strategies) {
    spawnActionContextMapsBuilder
        .remoteDynamicStrategyByMnemonicMap()
        .replaceValues(mnemonic, strategies);
    return this;
  }

  /**
   * Sets the strategy names to use in the local branch of dynamic execution for a given action
   * mnemonic.
   *
   * <p>During execution, each strategy is {@linkplain SpawnActionContext#canExec(Spawn,
   * com.google.devtools.build.lib.actions.ActionExecutionContext) asked} whether it can execute a
   * given Spawn. The first strategy in the list that says so will get the job.
   */
  public ExecutorBuilder addDynamicLocalStrategiesByMnemonic(
      String mnemonic, List<String> strategies) {
    spawnActionContextMapsBuilder
        .localDynamicStrategyByMnemonicMap()
        .replaceValues(mnemonic, strategies);
    return this;
  }

  /** Sets the strategy name to use if remote execution is not possible. */
  public ExecutorBuilder setRemoteFallbackStrategy(String remoteLocalFallbackStrategy) {
    spawnActionContextMapsBuilder.setRemoteFallbackStrategy(remoteLocalFallbackStrategy);
    return this;
  }

  /**
   * Adds an implementation with a specific strategy name.
   *
   * <p>Modules are free to provide different implementations of {@code ActionContext}. This can be
   * used, for example, to implement sandboxed or distributed execution of {@code SpawnAction}s in
   * different ways, while giving the user control over how exactly they are executed.
   *
   * <p>Example: a module requires {@code MyCustomActionContext} to be available, but doesn't
   * associate it with any strategy. Call <code>
   * addStrategyByContext(MyCustomActionContext.class, "")</code>.
   *
   * <p>Example: a module requires {@code MyLocalCustomActionContext} to be available, and wants it
   * to always use the "local" strategy. Call <code>
   * addStrategyByContext(MyCustomActionContext.class, "local")</code>.
   */
  public ExecutorBuilder addStrategyByContext(
      Class<? extends ActionContext> actionContext, String strategy) {
    spawnActionContextMapsBuilder.strategyByContextMap().put(actionContext, strategy);
    return this;
  }

  /**
   * Similar to {@link #addStrategyByMnemonic}, but allows specifying a regex for the set of
   * matching mnemonics, instead of an exact string.
   */
  public ExecutorBuilder addStrategyByRegexp(RegexFilter regexFilter, List<String> strategy) {
    spawnActionContextMapsBuilder.addStrategyByRegexp(regexFilter, strategy);
    return this;
  }

  /**
   * Sets the action input prefetcher. Only one module may set the prefetcher. If multiple modules
   * set it, this method will throw an {@link IllegalStateException}.
   */
  public ExecutorBuilder setActionInputPrefetcher(ActionInputPrefetcher prefetcher) {
    Preconditions.checkState(this.prefetcher == null);
    this.prefetcher = Preconditions.checkNotNull(prefetcher);
    return this;
  }
}
