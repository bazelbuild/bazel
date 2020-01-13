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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.RegexFilter;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Container for looking up the {@link ActionContext} to use for a given action.
 *
 * <p>Holds {@link ActionContext} mappings populated by modules. These include mappings from
 * mnemonics and from description patterns.
 *
 * <p>At startup time, the application provides {@link Builder} to each module to register its
 * contexts and mappings. At runtime, the {@link BlazeExecutor} uses the constructed object to find
 * the context for each action.
 */
public final class SpawnActionContextMaps
    implements DynamicStrategyRegistry, RemoteLocalFallbackRegistry {

  /** A stored entry for a {@link RegexFilter} to {@link SpawnActionContext} mapping. */
  @AutoValue
  public abstract static class RegexFilterSpawnActionContext {
    public abstract RegexFilter regexFilter();

    public abstract ImmutableList<SpawnActionContext> spawnActionContext();
  }

  private final ImmutableSortedMap<String, List<SpawnActionContext>> mnemonicToSpawnStrategiesMap;
  private final ImmutableList<ActionContext> strategies;
  private final ImmutableList<RegexFilterSpawnActionContext> spawnStrategyRegexList;
  private final ImmutableMultimap<String, SandboxedSpawnActionContext>
      mnemonicToRemoteDynamicStrategies;
  private final ImmutableMultimap<String, SandboxedSpawnActionContext>
      mnemonicToLocalDynamicStrategies;
  @Nullable private final AbstractSpawnStrategy remoteLocalFallbackStrategy;

  private SpawnActionContextMaps(
      ImmutableSortedMap<String, List<SpawnActionContext>> mnemonicToSpawnStrategiesMap,
      ImmutableList<ActionContext> strategies,
      ImmutableList<RegexFilterSpawnActionContext> spawnStrategyRegexList,
      ImmutableMultimap<String, SandboxedSpawnActionContext> mnemonicToRemoteDynamicStrategies,
      ImmutableMultimap<String, SandboxedSpawnActionContext> mnemonicToLocalDynamicStrategies,
      AbstractSpawnStrategy remoteLocalFallbackStrategy) {
    this.mnemonicToSpawnStrategiesMap = mnemonicToSpawnStrategiesMap;
    this.strategies = strategies;
    this.spawnStrategyRegexList = spawnStrategyRegexList;
    this.mnemonicToRemoteDynamicStrategies = mnemonicToRemoteDynamicStrategies;
    this.mnemonicToLocalDynamicStrategies = mnemonicToLocalDynamicStrategies;
    this.remoteLocalFallbackStrategy = remoteLocalFallbackStrategy;
  }

  /**
   * Returns a list of appropriate {@link ActionContext}s to execute the given {@link Spawn} with.
   *
   * <p>If the reason for selecting the context is worth mentioning to the user, logs a message
   * using the given {@link Reporter}.
   */
  List<SpawnActionContext> getSpawnActionContexts(Spawn spawn, EventHandler reporter) {
    Preconditions.checkNotNull(spawn);
    if (!spawnStrategyRegexList.isEmpty() && spawn.getResourceOwner() != null
            // Don't override test strategies by --strategy_regexp for backwards compatibility.
            && !"TestRunner".equals(spawn.getMnemonic())) {
      String description = spawn.getResourceOwner().getProgressMessage();
      if (description != null) {
        for (RegexFilterSpawnActionContext entry : spawnStrategyRegexList) {
          if (entry.regexFilter().isIncluded(description) && entry.spawnActionContext() != null) {
            reporter.handle(
                Event.progress(
                    description + " with context " + entry.spawnActionContext().toString()));
            return entry.spawnActionContext();
          }
        }
      }
    }
    List<SpawnActionContext> context = mnemonicToSpawnStrategiesMap.get(spawn.getMnemonic());
    if (context != null) {
      return context;
    }
    return Preconditions.checkNotNull(mnemonicToSpawnStrategiesMap.get(""));
  }

  @Override
  public List<SandboxedSpawnActionContext> getDynamicSpawnActionContexts(
      Spawn spawn, DynamicMode dynamicMode) {
    ImmutableMultimap<String, SandboxedSpawnActionContext> mnemonicToDynamicStrategies =
        dynamicMode == DynamicStrategyRegistry.DynamicMode.REMOTE
            ? mnemonicToRemoteDynamicStrategies
            : mnemonicToLocalDynamicStrategies;
    return ImmutableList.<SandboxedSpawnActionContext>builder()
        .addAll(mnemonicToDynamicStrategies.get(spawn.getMnemonic()))
        .addAll(mnemonicToDynamicStrategies.get(""))
        .build();
  }

  @Nullable
  @Override
  public AbstractSpawnStrategy getRemoteLocalFallbackStrategy() {
    return remoteLocalFallbackStrategy;
  }

  /** Returns a map from action context class to its instantiated context object. */
  @VisibleForTesting
  public ImmutableMap<Class<? extends ActionContext>, ActionContext> contextMap() {
    Map<Class<? extends ActionContext>, ActionContext> contextMap = new HashMap<>();
    for (ActionContext context : strategies) {
      ExecutionStrategy annotation = context.getClass().getAnnotation(ExecutionStrategy.class);
      if (annotation != null) {
        contextMap.put(annotation.contextType(), context);
      }
      contextMap.put(context.getClass(), context);
    }
    contextMap.put(SpawnActionContext.class, new ProxySpawnActionContext(this));
    contextMap.put(DynamicStrategyRegistry.class, this);
    contextMap.put(RemoteLocalFallbackRegistry.class, this);
    return ImmutableMap.copyOf(contextMap);
  }

  /** Returns a list of all referenced {@link ActionContext} instances. */
  @VisibleForTesting
  public ImmutableList<ActionContext> allContexts() {
    // We need to keep only the last occurrences of the entries in contextImplementations
    // (so we respect insertion order but also instantiate them only once).
    LinkedHashSet<ActionContext> allContexts = new LinkedHashSet<>();
    allContexts.addAll(strategies);
    mnemonicToSpawnStrategiesMap.values().forEach(allContexts::addAll);
    spawnStrategyRegexList.forEach(x -> allContexts.addAll(x.spawnActionContext()));
    return ImmutableList.copyOf(allContexts);
  }

  /**
   * Print a sorted list of our (Spawn)ActionContext maps.
   *
   * <p>Prints out debug information about the mappings.
   */
  void debugPrintSpawnActionContextMaps(Reporter reporter) {
    for (Entry<String, List<SpawnActionContext>> entry : mnemonicToSpawnStrategiesMap.entrySet()) {
      List<String> strategyNames =
          entry.getValue().stream()
              .map(spawnActionContext -> spawnActionContext.getClass().getSimpleName())
              .collect(Collectors.toList());
      reporter.handle(
          Event.info(
              String.format(
                  "SpawnActionContextMap: \"%s\" = [%s]",
                  entry.getKey(), Joiner.on(", ").join(strategyNames))));
    }

    ImmutableMap<Class<? extends ActionContext>, ActionContext> contextMap = contextMap();
    TreeMap<String, String> sortedContextMapWithSimpleNames = new TreeMap<>();
    for (Map.Entry<Class<? extends ActionContext>, ActionContext> entry : contextMap.entrySet()) {
      sortedContextMapWithSimpleNames.put(
          entry.getKey().getSimpleName(), entry.getValue().getClass().getSimpleName());
    }
    for (Map.Entry<String, String> entry : sortedContextMapWithSimpleNames.entrySet()) {
      // Skip uninteresting identity mappings of contexts.
      if (!entry.getKey().equals(entry.getValue())) {
        reporter.handle(
            Event.info(String.format("ContextMap: %s = %s", entry.getKey(), entry.getValue())));
      }
    }

    for (RegexFilterSpawnActionContext entry : spawnStrategyRegexList) {
      reporter.handle(
          Event.info(
              String.format(
                  "SpawnActionContextMap: \"%s\" = %s",
                  entry.regexFilter().toString(),
                  entry.spawnActionContext().getClass().getSimpleName())));
    }
  }

  @VisibleForTesting
  public static SpawnActionContextMaps createStub(
      List<ActionContext> strategies,
      Map<String, List<SpawnActionContext>> spawnStrategyMnemonicMap) {
    return new SpawnActionContextMaps(
        ImmutableSortedMap.copyOf(spawnStrategyMnemonicMap, String.CASE_INSENSITIVE_ORDER),
        ImmutableList.copyOf(strategies),
        ImmutableList.of(),
        ImmutableMultimap.of(),
        ImmutableMultimap.of(),
        /* remoteLocalFallbackStrategy=*/ null);
  }

  /** A stored entry for a {@link RegexFilter} to {@code strategy} mapping. */
  @AutoValue
  public abstract static class RegexFilterStrategy {
    public abstract RegexFilter regexFilter();

    public abstract ImmutableList<String> strategy();
  }

  /** Builder for {@code SpawnActionContextMaps}. */
  public static final class Builder {
    private final LinkedHashMultimap<String, String> strategyByMnemonicMap =
        LinkedHashMultimap.create();
    private ImmutableListMultimap.Builder<Class<? extends ActionContext>, String>
        strategyByContextMapBuilder = ImmutableListMultimap.builder();
    private final ImmutableList.Builder<RegexFilterStrategy> strategyByRegexpBuilder =
        ImmutableList.builder();
    private final LinkedHashMultimap<String, String> remoteDynamicStrategyByMnemonicMap =
        LinkedHashMultimap.create();
    private final LinkedHashMultimap<String, String> localDynamicStrategyByMnemonicMap =
        LinkedHashMultimap.create();
    @Nullable private String remoteLocalFallbackStrategyName;

    /**
     * Returns a builder modules can use to add mappings from mnemonics to strategy names.
     *
     * <p>If a spawn action is executed whose mnemonic maps to the empty string or is not present in
     * the map at all, the choice of the implementation is left to Blaze.
     *
     * <p>Matching on mnemonics is done case-insensitively so it is recommended that any module
     * makes sure that no two strategies refer to the same mnemonic. If they do, Blaze will pick the
     * last one added.
     */
    public LinkedHashMultimap<String, String> strategyByMnemonicMap() {
      return strategyByMnemonicMap;
    }

    /**
     * Returns a builder modules can use to add mappings from mnemonics to strategy names for use in
     * the remote branch of dynamic execution.
     *
     * <p>If a spawn action is executed whose mnemonic maps to the empty string or is not present in
     * the map at all, the choice of the implementation is left to Blaze.
     *
     * <p>Matching on mnemonics is done case-insensitively so it is recommended that any module
     * makes sure that no two strategies refer to the same mnemonic. If they do, Blaze will pick the
     * last one added.
     */
    public LinkedHashMultimap<String, String> remoteDynamicStrategyByMnemonicMap() {
      return remoteDynamicStrategyByMnemonicMap;
    }

    /**
     * Returns a builder modules can use to add mappings from mnemonics to strategy names for use in
     * the local branch of dynamic execution.
     *
     * <p>If a spawn action is executed whose mnemonic maps to the empty string or is not present in
     * the map at all, the choice of the implementation is left to Blaze.
     *
     * <p>Matching on mnemonics is done case-insensitively so it is recommended that any module
     * makes sure that no two strategies refer to the same mnemonic. If they do, Blaze will pick the
     * last one added.
     */
    public LinkedHashMultimap<String, String> localDynamicStrategyByMnemonicMap() {
      return localDynamicStrategyByMnemonicMap;
    }

    /**
     * Sets the command-line identifier of the strategy to be used when falling back from remote to
     * local execution.
     *
     * <p>Note that this is an optional setting, if not provided {@link
     * SpawnActionContextMaps#getRemoteLocalFallbackStrategy()} will return {@code null}. If the
     * value <b>is</b> provided it must match the commandline identifier of a registered strategy
     * (at {@linkplain #build build} time).
     */
    public void setRemoteFallbackStrategy(String remoteLocalFallbackStrategy) {
      this.remoteLocalFallbackStrategyName = remoteLocalFallbackStrategy;
    }

    /**
     * Returns a builder modules can use to associate {@link ActionContext} classes with strategy
     * names.
     */
    public ImmutableMultimap.Builder<Class<? extends ActionContext>, String>
        strategyByContextMap() {
      return strategyByContextMapBuilder;
    }

    /** Adds a mapping from the given {@link RegexFilter} to a {@code strategy}. */
    public void addStrategyByRegexp(RegexFilter regexFilter, List<String> strategy) {
      strategyByRegexpBuilder.add(
          new AutoValue_SpawnActionContextMaps_RegexFilterStrategy(
              regexFilter, ImmutableList.copyOf(strategy)));
    }

    /** Builds a {@link SpawnActionContextMaps} instance. */
    public SpawnActionContextMaps build(ImmutableList<ActionContextProvider> actionContextProviders)
        throws ExecutorInitException {
      StrategyConverter strategyConverter = new StrategyConverter(actionContextProviders);

      ImmutableSortedMap.Builder<String, List<SpawnActionContext>> spawnStrategyMap =
          ImmutableSortedMap.orderedBy(String.CASE_INSENSITIVE_ORDER);
      ImmutableList.Builder<ActionContext> strategies = ImmutableList.builder();
      ImmutableList.Builder<RegexFilterSpawnActionContext> spawnStrategyRegexList =
          ImmutableList.builder();

      for (String mnemonic : strategyByMnemonicMap.keySet()) {
        ImmutableList.Builder<SpawnActionContext> contexts = ImmutableList.builder();
        Set<String> strategiesForMnemonic = strategyByMnemonicMap.get(mnemonic);
        for (String strategy : strategiesForMnemonic) {
          SpawnActionContext context =
              strategyConverter.getStrategy(SpawnActionContext.class, strategy);
          if (context == null) {
            String strategyOrNull = Strings.emptyToNull(strategy);
            throw makeExceptionForInvalidStrategyValue(
                strategy,
                Joiner.on(' ').skipNulls().join(strategyOrNull, "spawn"),
                strategyConverter.getValidValues(SpawnActionContext.class));
          }
          contexts.add(context);
        }
        spawnStrategyMap.put(mnemonic, contexts.build());
      }

      Set<ActionContext> seenContext = new HashSet<>();
      for (Map.Entry<Class<? extends ActionContext>, String> entry :
          strategyByContextMapBuilder.orderValuesBy(Collections.reverseOrder()).build().entries()) {
        ActionContext context = strategyConverter.getStrategy(entry.getKey(), entry.getValue());
        if (context == null) {
          throw makeExceptionForInvalidStrategyValue(
              entry.getValue(),
              strategyConverter.getUserFriendlyName(entry.getKey()),
              strategyConverter.getValidValues(entry.getKey()));
        }
        if (seenContext.contains(context)) {
          continue;
        }
        seenContext.add(context);
        strategies.add(context);
      }

      for (RegexFilterStrategy entry : strategyByRegexpBuilder.build()) {
        ImmutableList.Builder<SpawnActionContext> contexts = ImmutableList.builder();
        List<String> strategiesForRegex = entry.strategy();
        for (String strategy : strategiesForRegex) {
          SpawnActionContext context =
              strategyConverter.getStrategy(SpawnActionContext.class, strategy);
          if (context == null) {
            strategy = Strings.emptyToNull(strategy);
            throw makeExceptionForInvalidStrategyValue(
                entry.regexFilter().toString(),
                Joiner.on(' ').skipNulls().join(strategy, "spawn"),
                strategyConverter.getValidValues(SpawnActionContext.class));
          }
          contexts.add(context);
        }
        spawnStrategyRegexList.add(
            new AutoValue_SpawnActionContextMaps_RegexFilterSpawnActionContext(
                entry.regexFilter(), contexts.build()));
      }

      AbstractSpawnStrategy remoteLocalFallbackStrategy = null;
      if (remoteLocalFallbackStrategyName != null) {
        SpawnActionContext strategy =
            strategyConverter.getStrategy(
                SpawnActionContext.class, remoteLocalFallbackStrategyName);
        if (!(strategy instanceof AbstractSpawnStrategy)) {
          throw makeExceptionForInvalidStrategyValue(
              remoteLocalFallbackStrategyName,
              "remote local fallback",
              strategyConverter.getValidValues(SpawnActionContext.class, "remote"));
        }
        remoteLocalFallbackStrategy = (AbstractSpawnStrategy) strategy;
      }

      return new SpawnActionContextMaps(
          spawnStrategyMap.build(),
          strategies.build(),
          spawnStrategyRegexList.build(),
          toActionContexts(strategyConverter, remoteDynamicStrategyByMnemonicMap),
          toActionContexts(strategyConverter, localDynamicStrategyByMnemonicMap),
          remoteLocalFallbackStrategy);
    }

    private ImmutableMultimap<String, SandboxedSpawnActionContext> toActionContexts(
        StrategyConverter strategyConverter,
        LinkedHashMultimap<String, String> dynamicStrategyByMnemonicMap)
        throws ExecutorInitException {
      ImmutableMultimap.Builder<String, SandboxedSpawnActionContext> mnemonicToStrategies =
          ImmutableMultimap.builder();
      for (Entry<String, Collection<String>> mnemonicToIdentifiers :
          dynamicStrategyByMnemonicMap.asMap().entrySet()) {
        for (String identifier : mnemonicToIdentifiers.getValue()) {
          if (identifier.isEmpty()) {
            continue;
          }
          SpawnActionContext context =
              strategyConverter.getStrategy(SpawnActionContext.class, identifier);
          if (context == null) {
            throw makeExceptionForInvalidStrategyValue(
                identifier,
                Joiner.on(' ').skipNulls().join(Strings.emptyToNull(identifier), "spawn"),
                strategyConverter.getValidValues(SpawnActionContext.class));
          }
          if (!(context instanceof SandboxedSpawnActionContext)) {
            throw new ExecutorInitException(
                "Requested strategy " + identifier + " exists but does not support sandboxing");
          }
          mnemonicToStrategies.put(
              mnemonicToIdentifiers.getKey(), (SandboxedSpawnActionContext) context);
        }
      }
      return mnemonicToStrategies.build();
    }
  }

  private static ExecutorInitException makeExceptionForInvalidStrategyValue(
      String value, String strategy, String validValues) {
    return new ExecutorInitException(
        String.format(
            "'%s' is an invalid value for %s strategy. Valid values are: %s",
            value, strategy, validValues),
        ExitCode.COMMAND_LINE_ERROR);
  }

  private static class StrategyConverter {
    private Table<Class<? extends ActionContext>, String, ActionContext> classMap =
        HashBasedTable.create();
    private Map<Class<? extends ActionContext>, ActionContext> defaultClassMap = new HashMap<>();

    /** Aggregates all {@link ActionContext}s that are in {@code contextProviders}. */
    @SuppressWarnings("unchecked")
    private StrategyConverter(Iterable<ActionContextProvider> contextProviders) {
      for (ActionContextProvider provider : contextProviders) {
        for (ActionContext strategy : provider.getActionContexts()) {
          ExecutionStrategy annotation = strategy.getClass().getAnnotation(ExecutionStrategy.class);
          // TODO(ulfjack): Don't silently ignore action contexts without annotation.
          if (annotation != null) {
            defaultClassMap.put(annotation.contextType(), strategy);

            for (String name : annotation.name()) {
              classMap.put(annotation.contextType(), name, strategy);
            }
          }
        }
      }
    }

    @SuppressWarnings("unchecked")
    private <T extends ActionContext> T getStrategy(Class<T> clazz, String name) {
      return (T) (name.isEmpty() ? defaultClassMap.get(clazz) : classMap.get(clazz, name));
    }

    private String getValidValues(Class<? extends ActionContext> context, String... excludes) {
      ImmutableSet<String> excludedNames = ImmutableSet.copyOf(excludes);
      return classMap.row(context).keySet().stream()
          .filter(s -> !excludedNames.contains(s))
          .sorted()
          .collect(Collectors.joining(", "));
    }

    private String getUserFriendlyName(Class<? extends ActionContext> context) {
      ActionContextMarker marker = context.getAnnotation(ActionContextMarker.class);
      return marker != null ? marker.name() : context.getSimpleName();
    }
  }
}
