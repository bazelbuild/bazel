// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.SandboxedSpawnStrategy;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.RegexFilter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Registry that collects spawn strategies and rules about their applicability and makes them
 * available for querying through various registry interfaces.
 *
 * <p>An instance of this registry can be created using its {@linkplain Builder builder}, which is
 * available to Blaze modules during server startup.
 */
public final class SpawnStrategyRegistry
    implements DynamicStrategyRegistry, ActionContext, RemoteLocalFallbackRegistry {

  private final ImmutableListMultimap<String, SpawnStrategy> mnemonicToStrategies;
  private final ImmutableListMultimap<RegexFilter, SpawnStrategy> filterToStrategies;
  private final ImmutableList<? extends SpawnStrategy> defaultStrategies;
  private final ImmutableMultimap<String, SandboxedSpawnStrategy> mnemonicToRemoteDynamicStrategies;
  private final ImmutableMultimap<String, SandboxedSpawnStrategy> mnemonicToLocalDynamicStrategies;
  @Nullable private final AbstractSpawnStrategy remoteLocalFallbackStrategy;

  private SpawnStrategyRegistry(
      ImmutableListMultimap<String, SpawnStrategy> mnemonicToStrategies,
      ImmutableListMultimap<RegexFilter, SpawnStrategy> filterToStrategies,
      ImmutableList<? extends SpawnStrategy> defaultStrategies,
      ImmutableMultimap<String, SandboxedSpawnStrategy> mnemonicToRemoteDynamicStrategies,
      ImmutableMultimap<String, SandboxedSpawnStrategy> mnemonicToLocalDynamicStrategies,
      @Nullable AbstractSpawnStrategy remoteLocalFallbackStrategy) {
    this.mnemonicToStrategies = mnemonicToStrategies;
    this.filterToStrategies = filterToStrategies;
    this.defaultStrategies = defaultStrategies;
    this.mnemonicToRemoteDynamicStrategies = mnemonicToRemoteDynamicStrategies;
    this.mnemonicToLocalDynamicStrategies = mnemonicToLocalDynamicStrategies;
    this.remoteLocalFallbackStrategy = remoteLocalFallbackStrategy;
  }

  /**
   * Returns the strategies applying to the given spawn, in priority order.
   *
   * <p>Which strategies are returned is based on the precedence as documented on the construction
   * methods of {@linkplain Builder this registry's builder}.
   *
   * <p>If the reason for selecting the context is worth mentioning to the user, logs a message
   * using the given {@link Reporter}.
   */
  @VisibleForTesting
  public List<? extends SpawnStrategy> getStrategies(Spawn spawn, EventHandler reporter) {
    // Don't override test strategies by --strategy_regexp for backwards compatibility.
    if (spawn.getResourceOwner() != null && !"TestRunner".equals(spawn.getMnemonic())) {
      String description = spawn.getResourceOwner().getProgressMessage();
      if (description != null) {
        for (Map.Entry<RegexFilter, Collection<SpawnStrategy>> filterStrategies :
            filterToStrategies.asMap().entrySet()) {
          if (filterStrategies.getKey().isIncluded(description)) {
            // TODO(schmitt): Why is this done here and not after running canExec?
            reporter.handle(
                Event.progress(description + " with context " + filterStrategies.getValue()));
            return ImmutableList.copyOf(filterStrategies.getValue());
          }
        }
      }
    }
    if (mnemonicToStrategies.containsKey(spawn.getMnemonic())) {
      return mnemonicToStrategies.get(spawn.getMnemonic());
    }
    return defaultStrategies;
  }

  @Override
  public void notifyUsedDynamic(ActionContext.ActionContextRegistry actionContextRegistry) {
    for (SandboxedSpawnStrategy strategy : mnemonicToLocalDynamicStrategies.values()) {
      strategy.usedContext(actionContextRegistry);
    }
    for (SandboxedSpawnStrategy strategy : mnemonicToRemoteDynamicStrategies.values()) {
      strategy.usedContext(actionContextRegistry);
    }
  }

  @Override
  public List<SandboxedSpawnStrategy> getDynamicSpawnActionContexts(
      Spawn spawn, DynamicMode dynamicMode) {
    ImmutableMultimap<String, SandboxedSpawnStrategy> mnemonicToDynamicStrategies =
        dynamicMode == DynamicStrategyRegistry.DynamicMode.REMOTE
            ? mnemonicToRemoteDynamicStrategies
            : mnemonicToLocalDynamicStrategies;
    return ImmutableList.<SandboxedSpawnStrategy>builder()
        .addAll(mnemonicToDynamicStrategies.get(spawn.getMnemonic()))
        .addAll(mnemonicToDynamicStrategies.get(""))
        .build();
  }

  @Nullable
  @Override
  public AbstractSpawnStrategy getRemoteLocalFallbackStrategy() {
    return remoteLocalFallbackStrategy;
  }

  /**
   * Notifies all (non-dynamic) strategies stored in this registry that they are {@linkplain
   * SpawnStrategy#usedContext used}.
   */
  public void notifyUsed(ActionContext.ActionContextRegistry actionContextRegistry) {
    for (SpawnStrategy strategy : filterToStrategies.values()) {
      strategy.usedContext(actionContextRegistry);
    }
    for (SpawnStrategy strategy : mnemonicToStrategies.values()) {
      strategy.usedContext(actionContextRegistry);
    }
    for (SpawnStrategy strategy : defaultStrategies) {
      strategy.usedContext(actionContextRegistry);
    }
    if (remoteLocalFallbackStrategy != null) {
      remoteLocalFallbackStrategy.usedContext(actionContextRegistry);
    }
  }

  /**
   * Records the list of all spawn strategies that can be returned by the various query methods of
   * this registry to the given reporter.
   */
  void writeSpawnStrategiesTo(Reporter reporter) {
    for (Map.Entry<String, Collection<SpawnStrategy>> entry :
        mnemonicToStrategies.asMap().entrySet()) {
      reporter.handle(
          Event.info(
              String.format(
                  "MnemonicToStrategyImplementations: \"%s\" = [%s]",
                  entry.getKey(), toImplementationNames(entry.getValue()))));
    }

    for (Map.Entry<RegexFilter, Collection<SpawnStrategy>> entry :
        filterToStrategies.asMap().entrySet()) {
      Collection<SpawnStrategy> value = entry.getValue();
      reporter.handle(
          Event.info(
              String.format(
                  "FilterToStrategyImplementations: \"%s\" = [%s]",
                  entry.getKey(), toImplementationNames(value))));
    }

    reporter.handle(
        Event.info(
            String.format(
                "DefaultStrategyImplementations: [%s]", toImplementationNames(defaultStrategies))));

    if (remoteLocalFallbackStrategy != null) {
      reporter.handle(
          Event.info(
              String.format(
                  "RemoteLocalFallbackImplementation: [%s]",
                  remoteLocalFallbackStrategy.getClass().getSimpleName())));
    }

    for (Map.Entry<String, Collection<SandboxedSpawnStrategy>> entry :
        mnemonicToRemoteDynamicStrategies.asMap().entrySet()) {
      reporter.handle(
          Event.info(
              String.format(
                  "MnemonicToRemoteDynamicStrategyImplementations: \"%s\" = [%s]",
                  entry.getKey(), toImplementationNames(entry.getValue()))));
    }

    for (Map.Entry<String, Collection<SandboxedSpawnStrategy>> entry :
        mnemonicToLocalDynamicStrategies.asMap().entrySet()) {
      reporter.handle(
          Event.info(
              String.format(
                  "MnemonicToLocalDynamicStrategyImplementations: \"%s\" = [%s]",
                  entry.getKey(), toImplementationNames(entry.getValue()))));
    }
  }

  private String toImplementationNames(Collection<?> strategies) {
    return strategies.stream()
        .map(strategy -> strategy.getClass().getSimpleName())
        .collect(joining(", "));
  }

  /** Returns a new {@link Builder} suitable for creating instances of SpawnStrategyRegistry. */
  public static Builder builder() {
    return new BuilderImpl();
  }

  /**
   * Builder collecting the strategies and restrictions thereon for a {@link SpawnStrategyRegistry}.
   *
   * <p>To {@linkplain SpawnStrategyRegistry#getStrategies match a strategy to a spawn} it needs to
   * be both {@linkplain #registerStrategy registered} and its registered command-line identifier
   * has to match {@linkplain #addDescriptionFilter a filter on the spawn's progress message},
   * {@linkplain #addMnemonicFilter a filter on the spawn's mnemonic} or be part of the default
   * strategies (see below).
   *
   * <p><strong>Default strategies</strong> are either {@linkplain #setDefaultStrategies set
   * explicitly} or, if {@link #setDefaultStrategies} is not called on this builder, comprised of
   * all registered strategies, in registration order (i.e. the earliest strategy registered will be
   * first in the list of strategies returned by {@link SpawnStrategyRegistry#getStrategies}).
   */
  // TODO(katre): This exists only to allow incremental migration from SpawnActionContextMaps.
  // Delete ASAP.
  public interface Builder {

    /**
     * Adds a filter limiting any spawn whose {@linkplain
     * ActionExecutionMetadata#getProgressMessage() owner's progress message} matches the regular
     * expression to only use strategies with the given command-line identifiers, in order.
     *
     * <p>If multiple filters match the same spawn (including an identical filter) the order of
     * precedence of calls to this method is determined by {@link
     * #useLegacyDescriptionFilterPrecedence()}.
     */
    SpawnStrategyRegistry.Builder addDescriptionFilter(
        RegexFilter filter, List<String> identifiers);

    /**
     * Adds a filter limiting any spawn whose {@linkplain Spawn#getMnemonic() mnemonic}
     * (case-sensitively) matches the given mnemonic to only use strategies with the given
     * command-line identifiers, in order.
     *
     * <p>If the same mnemonic is registered multiple times the last such call will take precedence.
     *
     * <p>Note that if a spawn matches a {@linkplain #addDescriptionFilter registered description
     * filter} that filter will take precedence over any mnemonic-based filters.
     */
    // last one wins
    SpawnStrategyRegistry.Builder addMnemonicFilter(String mnemonic, List<String> identifiers);

    default SpawnStrategyRegistry.Builder registerStrategy(
        SpawnStrategy strategy, String... commandlineIdentifiers) {
      return registerStrategy(strategy, ImmutableList.copyOf(commandlineIdentifiers));
    }

    /**
     * Registers a strategy implementation with this collector, distinguishing it from other
     * strategies with the given command-line identifiers (of which at least one is required).
     *
     * <p>If multiple strategies are registered with the same command-line identifier the last one
     * so registered will take precedence.
     */
    SpawnStrategyRegistry.Builder registerStrategy(
        SpawnStrategy strategy, List<String> commandlineIdentifiers);

    /**
     * Instructs this collector to use the legacy description filter precedence, i.e. to prefer the
     * first regular expression filter that matches a spawn over any later registered filters.
     *
     * <p>The default behavior of this collector is to prefer the last registered description filter
     * over any previously registered matching filters.
     */
    SpawnStrategyRegistry.Builder useLegacyDescriptionFilterPrecedence();

    /**
     * Explicitly sets the identifiers of default strategies to use if a spawn matches no filters.
     *
     * <p>Note that if this method is not called on the builder, all registered strategies are
     * considered default strategies, in registration order. See also the {@linkplain Builder class
     * documentation}.
     */
    SpawnStrategyRegistry.Builder setDefaultStrategies(List<String> defaultStrategies);

    /**
     * Reset the default strategies (see {@link #setDefaultStrategies}) to the reverse of the order
     * they were registered in.
     */
    SpawnStrategyRegistry.Builder resetDefaultStrategies();

    /**
     * Sets the strategy names to use in the remote branch of dynamic execution for a given action
     * mnemonic.
     *
     * <p>During execution, each strategy is {@linkplain SpawnStrategy#canExec(Spawn,
     * ActionContextRegistry) asked} whether it can execute a given Spawn. The first strategy in the
     * list that says so will get the job.
     */
    SpawnStrategyRegistry.Builder addDynamicRemoteStrategiesByMnemonic(
        String mnemonic, List<String> strategies);

    /**
     * Sets the strategy names to use in the local branch of dynamic execution for a given action
     * mnemonic.
     *
     * <p>During execution, each strategy is {@linkplain SpawnStrategy#canExec(Spawn,
     * ActionContextRegistry) asked} whether it can execute a given Spawn. The first strategy in the
     * list that says so will get the job.
     */
    SpawnStrategyRegistry.Builder addDynamicLocalStrategiesByMnemonic(
        String mnemonic, List<String> strategies);

    /**
     * Sets the commandline identifier of the strategy to be used when falling back from remote to
     * local execution.
     *
     * <p>Note that this is an optional setting, if not provided {@link
     * SpawnStrategyRegistry#getRemoteLocalFallbackStrategy()} will return {@code null}. If the
     * value <b>is</b> provided it must match the commandline identifier of a registered strategy
     * (at {@linkplain #build build} time).
     */
    SpawnStrategyRegistry.Builder setRemoteLocalFallbackStrategyIdentifier(
        String commandlineIdentifier);

    /**
     * Finalizes the construction of the registry.
     *
     * @throws ExecutorInitException if a strategy command-line identifier was used in a filter or
     *     the default strategies but no strategy for that identifier was registered
     */
    SpawnStrategyRegistry build() throws ExecutorInitException;
  }

  /**
   * Builder collecting the strategies and restrictions thereon for a {@link SpawnStrategyRegistry}.
   *
   * <p>To {@linkplain SpawnStrategyRegistry#getStrategies match a strategy to a spawn} it needs to
   * be both {@linkplain #registerStrategy registered} and its registered command-line identifier
   * has to match {@linkplain #addDescriptionFilter a filter on the spawn's progress message},
   * {@linkplain #addMnemonicFilter a filter on the spawn's mnemonic} or be part of the default
   * strategies (see below).
   *
   * <p><strong>Default strategies</strong> are either {@linkplain #setDefaultStrategies set
   * explicitly} or, if {@link #setDefaultStrategies} is not called on this builder, comprised of
   * all registered strategies, in registration order (i.e. the earliest strategy registered will be
   * first in the list of strategies returned by {@link SpawnStrategyRegistry#getStrategies}).
   */
  private static final class BuilderImpl implements Builder {

    private ImmutableList<String> explicitDefaultStrategies = ImmutableList.of();
    // TODO(schmitt): Using a list and autovalue so as to be able to reverse order while legacy sort
    //  is supported. Can be converted to same as mnemonics once legacy behavior is removed.
    private final List<FilterAndIdentifiers> filterAndIdentifiers = new ArrayList<>();
    private final HashMap<String, SpawnStrategy> identifierToStrategy = new HashMap<>();
    private final ArrayList<SpawnStrategy> strategiesInRegistrationOrder = new ArrayList<>();

    // Using List values here rather than multimaps as there is no need for the latter's
    // functionality: The values are always replaced as a whole, no adding/creation required.
    private final HashMap<String, List<String>> mnemonicToIdentifiers = new HashMap<>();
    private final HashMap<String, List<String>> mnemonicToRemoteIdentifiers = new HashMap<>();
    private final HashMap<String, List<String>> mnemonicToLocalIdentifiers = new HashMap<>();
    private boolean legacyFilterIterationOrder = false;
    @Nullable private String remoteLocalFallbackStrategyIdentifier;

    /**
     * Adds a filter limiting any spawn whose {@linkplain
     * ActionExecutionMetadata#getProgressMessage() owner's progress message} matches the regular
     * expression to only use strategies with the given command-line identifiers, in order.
     *
     * <p>If multiple filters match the same spawn (including an identical filter) the order of
     * precedence of calls to this method is determined by {@link
     * #useLegacyDescriptionFilterPrecedence()}.
     */
    @Override
    public Builder addDescriptionFilter(RegexFilter filter, List<String> identifiers) {
      filterAndIdentifiers.add(
          new AutoValue_SpawnStrategyRegistry_FilterAndIdentifiers(
              filter, ImmutableList.copyOf(identifiers)));
      return this;
    }

    /**
     * Adds a filter limiting any spawn whose {@linkplain Spawn#getMnemonic() mnemonic}
     * (case-sensitively) matches the given mnemonic to only use strategies with the given
     * command-line identifiers, in order.
     *
     * <p>If the same mnemonic is registered multiple times the last such call will take precedence.
     *
     * <p>Note that if a spawn matches a {@linkplain #addDescriptionFilter registered description
     * filter} that filter will take precedence over any mnemonic-based filters.
     */
    // last one wins
    @Override
    public Builder addMnemonicFilter(String mnemonic, List<String> identifiers) {
      mnemonicToIdentifiers.put(mnemonic, identifiers);
      return this;
    }

    /**
     * Registers a strategy implementation with this collector, distinguishing it from other
     * strategies with the given command-line identifiers (of which at least one is required).
     *
     * <p>If multiple strategies are registered with the same command-line identifier the last one
     * so registered will take precedence.
     */
    @Override
    public Builder registerStrategy(SpawnStrategy strategy, List<String> commandlineIdentifiers) {
      Preconditions.checkArgument(
          commandlineIdentifiers.size() >= 1, "At least one commandLineIdentifier must be given");
      for (String identifier : commandlineIdentifiers) {
        identifierToStrategy.put(identifier, strategy);
      }
      strategiesInRegistrationOrder.add(strategy);
      return this;
    }

    /**
     * Instructs this collector to use the legacy description filter precedence, i.e. to prefer the
     * first regular expression filter that matches a spawn over any later registered filters.
     *
     * <p>The default behavior of this collector is to prefer the last registered description filter
     * over any previously registered matching filters.
     */
    @Override
    public Builder useLegacyDescriptionFilterPrecedence() {
      legacyFilterIterationOrder = true;
      return this;
    }

    /**
     * Explicitly sets the identifiers of default strategies to use if a spawn matches no filters.
     *
     * <p>Note that if this method is not called on the builder, all registered strategies are
     * considered default strategies, in registration order. See also the {@linkplain Builder class
     * documentation}.
     */
    @Override
    public Builder setDefaultStrategies(List<String> defaultStrategies) {
      // Ensure there are actual strategies and the contents are not empty.
      Preconditions.checkArgument(!defaultStrategies.isEmpty());
      Preconditions.checkArgument(
          defaultStrategies.stream().anyMatch(strategy -> !"".equals(strategy)));
      this.explicitDefaultStrategies = ImmutableList.copyOf(defaultStrategies);
      return this;
    }

    /**
     * Reset the default strategies (see {@link #setDefaultStrategies}) to the reverse of the order
     * they were registered in.
     */
    @Override
    public Builder resetDefaultStrategies() {
      this.explicitDefaultStrategies = ImmutableList.of();
      return this;
    }

    /**
     * Sets the strategy names to use in the remote branch of dynamic execution for a given action
     * mnemonic.
     *
     * <p>During execution, each strategy is {@linkplain SpawnStrategy#canExec(Spawn,
     * ActionContextRegistry) asked} whether it can execute a given Spawn. The first strategy in the
     * list that says so will get the job.
     */
    @Override
    public Builder addDynamicRemoteStrategiesByMnemonic(String mnemonic, List<String> strategies) {
      mnemonicToRemoteIdentifiers.put(mnemonic, strategies);
      return this;
    }

    /**
     * Sets the strategy names to use in the local branch of dynamic execution for a given action
     * mnemonic.
     *
     * <p>During execution, each strategy is {@linkplain SpawnStrategy#canExec(Spawn,
     * ActionContextRegistry) asked} whether it can execute a given Spawn. The first strategy in the
     * list that says so will get the job.
     */
    @Override
    public Builder addDynamicLocalStrategiesByMnemonic(String mnemonic, List<String> strategies) {
      mnemonicToLocalIdentifiers.put(mnemonic, strategies);
      return this;
    }

    /**
     * Sets the commandline identifier of the strategy to be used when falling back from remote to
     * local execution.
     *
     * <p>Note that this is an optional setting, if not provided {@link
     * SpawnStrategyRegistry#getRemoteLocalFallbackStrategy()} will return {@code null}. If the
     * value <b>is</b> provided it must match the commandline identifier of a registered strategy
     * (at {@linkplain #build build} time).
     */
    @Override
    public Builder setRemoteLocalFallbackStrategyIdentifier(String commandlineIdentifier) {
      this.remoteLocalFallbackStrategyIdentifier = commandlineIdentifier;
      return this;
    }

    /**
     * Finalizes the construction of the registry.
     *
     * @throws ExecutorInitException if a strategy command-line identifier was used in a filter or
     *     the default strategies but no strategy for that identifier was registered
     */
    @Override
    public SpawnStrategyRegistry build() throws ExecutorInitException {
      List<FilterAndIdentifiers> orderedFilterAndIdentifiers = filterAndIdentifiers;

      if (!legacyFilterIterationOrder) {
        orderedFilterAndIdentifiers = Lists.reverse(filterAndIdentifiers);
      }

      ListMultimap<RegexFilter, SpawnStrategy> filterToStrategies = LinkedListMultimap.create();
      for (FilterAndIdentifiers filterAndIdentifier : orderedFilterAndIdentifiers) {
        RegexFilter filter = filterAndIdentifier.filter();
        filterToStrategies.putAll(filter, toStrategies(filterAndIdentifier.identifiers(), filter));
      }

      ImmutableListMultimap.Builder<String, SpawnStrategy> mnemonicToStrategies =
          new ImmutableListMultimap.Builder<>();
      for (Map.Entry<String, List<String>> entry : mnemonicToIdentifiers.entrySet()) {
        mnemonicToStrategies.putAll(
            entry.getKey(), toStrategies(entry.getValue(), "mnemonic " + entry.getKey()));
      }

      ImmutableListMultimap.Builder<String, SandboxedSpawnStrategy> mnemonicToLocalStrategies =
          new ImmutableListMultimap.Builder<>();
      for (Map.Entry<String, List<String>> entry : mnemonicToLocalIdentifiers.entrySet()) {
        mnemonicToLocalStrategies.putAll(
            entry.getKey(),
            toSandboxedStrategies(entry.getValue(), "local mnemonic " + entry.getKey()));
      }

      ImmutableListMultimap.Builder<String, SandboxedSpawnStrategy> mnemonicToRemoteStrategies =
          new ImmutableListMultimap.Builder<>();
      for (Map.Entry<String, List<String>> entry : mnemonicToRemoteIdentifiers.entrySet()) {
        mnemonicToRemoteStrategies.putAll(
            entry.getKey(),
            toSandboxedStrategies(entry.getValue(), "remote mnemonic " + entry.getKey()));
      }

      AbstractSpawnStrategy remoteLocalFallbackStrategy = null;
      if (remoteLocalFallbackStrategyIdentifier != null) {
        SpawnStrategy strategy =
            toStrategy("remote fallback strategy", remoteLocalFallbackStrategyIdentifier);
        if (!(strategy instanceof AbstractSpawnStrategy)) {
          // TODO(schmitt): Check if all strategies can use the same base and remove check if so.
          throw new ExecutorInitException(
              String.format(
                  "'%s' was requested for the remote fallback strategy but is not an abstract "
                      + "spawn strategy (which is required for remote fallback execution).",
                  strategy.getClass().getSimpleName()),
              ExitCode.COMMAND_LINE_ERROR);
        }

        remoteLocalFallbackStrategy = (AbstractSpawnStrategy) strategy;
      }

      ImmutableList<? extends SpawnStrategy> defaultStrategies;
      if (explicitDefaultStrategies.isEmpty()) {
        // Use the strategies as registered, in reverse order.
        defaultStrategies = ImmutableList.copyOf(Lists.reverse(strategiesInRegistrationOrder));
      } else {
        defaultStrategies = toStrategies(explicitDefaultStrategies, "default strategies");
      }

      return new SpawnStrategyRegistry(
          mnemonicToStrategies.build(),
          ImmutableListMultimap.copyOf(filterToStrategies),
          defaultStrategies,
          mnemonicToRemoteStrategies.build(),
          mnemonicToLocalStrategies.build(),
          remoteLocalFallbackStrategy);
    }

    private ImmutableList<? extends SpawnStrategy> toStrategies(
        List<String> identifiers, Object requestName) throws ExecutorInitException {
      ImmutableList.Builder<SpawnStrategy> strategies = ImmutableList.builder();
      for (String identifier : identifiers) {
        if (identifier.isEmpty()) {
          continue;
        }
        strategies.add(toStrategy(requestName, identifier));
      }
      return strategies.build();
    }

    private SpawnStrategy toStrategy(Object requestName, String identifier)
        throws ExecutorInitException {
      SpawnStrategy strategy = identifierToStrategy.get(identifier);
      if (strategy == null) {
        throw new ExecutorInitException(
            String.format(
                "'%s' was requested for %s but no strategy with that identifier was registered. "
                    + "Valid values are: [%s]",
                identifier, requestName, Joiner.on(", ").join(identifierToStrategy.keySet())),
            ExitCode.COMMAND_LINE_ERROR);
      }
      return strategy;
    }

    private Iterable<? extends SandboxedSpawnStrategy> toSandboxedStrategies(
        List<String> identifiers, Object requestName) throws ExecutorInitException {
      Iterable<? extends SpawnStrategy> strategies = toStrategies(identifiers, requestName);
      for (SpawnStrategy strategy : strategies) {
        if (!(strategy instanceof SandboxedSpawnStrategy)) {
          throw new ExecutorInitException(
              String.format(
                  "'%s' was requested for %s but is not a sandboxed strategy (which is required for"
                      + " dynamic execution).",
                  strategy.getClass().getSimpleName(), requestName),
              ExitCode.COMMAND_LINE_ERROR);
        }
      }

      @SuppressWarnings("unchecked") // Each element of the iterable was checked to fulfil this.
      Iterable<? extends SandboxedSpawnStrategy> sandboxedStrategies =
          (Iterable<? extends SandboxedSpawnStrategy>) strategies;
      return sandboxedStrategies;
    }
  }

  @AutoValue
  abstract static class FilterAndIdentifiers {

    abstract RegexFilter filter();

    abstract ImmutableList<String> identifiers();
  }
}
