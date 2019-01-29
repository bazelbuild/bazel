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
   * Sets the strategy name for a given action mnemonic.
   *
   * <p>The calling module can either decide for itself which implementation is needed and make the
   * value associated with this key a constant or defer that decision to the user, for example, by
   * providing a command line option and setting the value in the map based on that.
   *
   * <p>Setting the strategy to the empty string "" redirects it to the value for the empty
   * mnemonic.
   *
   * <p>Example: a module requires {@code SpawnActionContext} to do its job, and it creates actions
   * with the mnemonic <code>C++</code>. The the module can call
   * <code>addStrategyByMnemonic("C++", strategy)</code>.
   */
  public ExecutorBuilder addStrategyByMnemonic(String mnemonic, String strategy) {
    spawnActionContextMapsBuilder.strategyByMnemonicMap().put(mnemonic, strategy);
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
   * associate it with any strategy. Call
   * <code>addStrategyByContext(MyCustomActionContext.class, "")</code>.
   *
   * <p>Example: a module requires {@code MyLocalCustomActionContext} to be available, and wants
   * it to always use the "local" strategy. Call
   * <code>addStrategyByContext(MyCustomActionContext.class, "local")</code>.
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
  public ExecutorBuilder addStrategyByRegexp(RegexFilter regexFilter, String strategy) {
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
