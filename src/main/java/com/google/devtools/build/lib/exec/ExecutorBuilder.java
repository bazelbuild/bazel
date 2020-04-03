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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.util.RegexFilter;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Builder class to create an {@link Executor} instance. This class is part of the module API,
 * which allows modules to affect how the executor is initialized.
 */
public class ExecutorBuilder {
  private final SpawnActionContextMaps.Builder spawnActionContextMapsBuilder =
      new SpawnActionContextMaps.Builder();
  private final Set<ExecutorLifecycleListener> executorLifecycleListeners = new LinkedHashSet<>();
  private ActionInputPrefetcher prefetcher;

  public SpawnActionContextMaps getSpawnActionContextMaps() throws ExecutorInitException {
    return spawnActionContextMapsBuilder.build();
  }

  /** Returns all executor lifecycle listeners registered with this builder so far. */
  public ImmutableSet<ExecutorLifecycleListener> getExecutorLifecycleListeners() {
    return ImmutableSet.copyOf(executorLifecycleListeners);
  }

  public ActionInputPrefetcher getActionInputPrefetcher() {
    return prefetcher == null ? ActionInputPrefetcher.NONE : prefetcher;
  }

  /**
   * Adds the specified action context to the executor, by wrapping it in a simple action context
   * provider implementation.
   *
   * <p>If two action contexts are registered that share an identifying type and commandline
   * identifier the last registered will take precedence.
   */
  public <T extends ActionContext> ExecutorBuilder addActionContext(
      Class<T> identifyingType, T context, String... commandlineIdentifiers) {
    spawnActionContextMapsBuilder.addContext(identifyingType, context, commandlineIdentifiers);
    return this;
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
   * <p>During execution, each strategy is {@linkplain SpawnStrategy#canExec(Spawn,
   * ActionContext.ActionContextRegistry) asked} whether it can execute a given Spawn. The first
   * strategy in the list that says so will get the job.
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
   * <p>During execution, each strategy is {@linkplain SpawnStrategy#canExec(Spawn,
   * ActionContext.ActionContextRegistry) asked} whether it can execute a given Spawn. The first
   * strategy in the list that says so will get the job.
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

  /**
   * Registers an executor lifecycle listener which will receive notifications throughout the
   * execution phase (if one occurs).
   *
   * @see ExecutorLifecycleListener for events that can be listened to
   */
  public ExecutorBuilder addExecutorLifecycleListener(ExecutorLifecycleListener listener) {
    executorLifecycleListeners.add(listener);
    return this;
  }

  // TODO(katre): Use a fake implementation to allow for migration to the new API.
  public ModuleActionContextRegistry.Builder asModuleActionContextRegistryBuilder() {
    return new ModuleActionContextDelegate(this);
  }

  private static final class ModuleActionContextDelegate
      implements ModuleActionContextRegistry.Builder {
    private final ExecutorBuilder executorBuilder;

    private ModuleActionContextDelegate(ExecutorBuilder executorBuilder) {
      this.executorBuilder = executorBuilder;
    }

    @Override
    public ModuleActionContextRegistry.Builder restrictTo(
        Class<?> identifyingType, String restriction) {
      Preconditions.checkArgument(ActionContext.class.isAssignableFrom(identifyingType));
      @SuppressWarnings("unchecked")
      Class<? extends ActionContext> castType = (Class<? extends ActionContext>) identifyingType;
      this.executorBuilder.addStrategyByContext(castType, restriction);
      return this;
    }

    @Override
    public <T extends ActionContext> ModuleActionContextRegistry.Builder register(
        Class<T> identifyingType, T context, String... commandLineIdentifiers) {
      this.executorBuilder.addActionContext(identifyingType, context, commandLineIdentifiers);
      return this;
    }

    @Override
    public ModuleActionContextRegistry build() throws ExecutorInitException {
      throw new UnsupportedOperationException("not a real builder");
    }
  }

  // TODO(katre): Use a fake implementation to allow for migration to the new API.
  public SpawnStrategyRegistry.Builder asSpawnStrategyRegistryBuilder() {
    return new SpawnStrategyRegistryDelegate(this);
  }

  private static final class SpawnStrategyRegistryDelegate
      implements SpawnStrategyRegistry.Builder {
    private final ExecutorBuilder executorBuilder;

    private SpawnStrategyRegistryDelegate(ExecutorBuilder executorBuilder) {
      this.executorBuilder = executorBuilder;
    }

    @Override
    public SpawnStrategyRegistry.Builder addDescriptionFilter(
        RegexFilter filter, List<String> identifiers) {
      this.executorBuilder.addStrategyByRegexp(filter, identifiers);
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder addMnemonicFilter(
        String mnemonic, List<String> identifiers) {
      this.executorBuilder.addStrategyByMnemonic(mnemonic, identifiers);
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder registerStrategy(
        SpawnStrategy strategy, List<String> commandlineIdentifiers) {
      this.executorBuilder.addActionContext(
          SpawnStrategy.class, strategy, commandlineIdentifiers.toArray(new String[0]));
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder useLegacyDescriptionFilterPrecedence() {
      // Ignored.
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder setDefaultStrategies(List<String> defaultStrategies) {
      this.executorBuilder.addStrategyByMnemonic("", defaultStrategies);
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder resetDefaultStrategies() {
      this.executorBuilder.addStrategyByMnemonic("", ImmutableList.of(""));
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder addDynamicRemoteStrategiesByMnemonic(
        String mnemonic, List<String> strategies) {
      this.executorBuilder.addDynamicRemoteStrategiesByMnemonic(mnemonic, strategies);
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder addDynamicLocalStrategiesByMnemonic(
        String mnemonic, List<String> strategies) {
      this.executorBuilder.addDynamicLocalStrategiesByMnemonic(mnemonic, strategies);
      return this;
    }

    @Override
    public SpawnStrategyRegistry.Builder setRemoteLocalFallbackStrategyIdentifier(
        String commandlineIdentifier) {
      this.executorBuilder.setRemoteFallbackStrategy(commandlineIdentifier);
      return this;
    }

    @Override
    public SpawnStrategyRegistry build() throws ExecutorInitException {
      throw new UnsupportedOperationException("not a real builder");
    }
  }
}
