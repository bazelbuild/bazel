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
import com.google.common.base.Joiner;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.common.collect.MutableClassToInstanceMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.ExitCode;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Registry containing all available {@linkplain ActionContext action contexts}.
 *
 * <p>Contexts can be {@linkplain #getContext queried} by a common subtype of {@link ActionContext}
 * that they implement (which can be the implementation class itself). It is possible to {@linkplain
 * Builder#restrictTo restrict) the available contexts for a type to those who were {@linkplain
 * Builder#register registered with specific command-line identifiers}. If more than one context was
 * {@link Builder#register registered} for the same type and they are not distinguished by the
 * restriction then this registry will return the last registered context.
 *
 * <p>An instance of this registry can be created using its {@linkplain Builder builder}, which is
 * available to Blaze modules during server startup.
 */
public final class ModuleActionContextRegistry
    implements ActionContext, ActionContext.ActionContextRegistry {

  private final ImmutableClassToInstanceMap<ActionContext> identifyingTypeToContext;

  private ModuleActionContextRegistry(
      ImmutableClassToInstanceMap<ActionContext> identifyingTypeToContext) {
    this.identifyingTypeToContext = identifyingTypeToContext;
  }

  @Override
  public <T extends ActionContext> T getContext(Class<T> identifyingType) {
    return identifyingTypeToContext.getInstance(identifyingType);
  }

  /**
   * Notifies all contexts stored in this registry that they are {@linkplain
   * ActionContext#usedContext used}.
   */
  public void notifyUsed() {
    for (ActionContext context : identifyingTypeToContext.values()) {
      context.usedContext(this);
    }
  }

  /**
   * Records the list of all contexts that can be {@linkplain #getContext returned by this registry}
   * to the given reporter.
   */
  void writeActionContextsTo(Reporter reporter) {
    for (Map.Entry<Class<? extends ActionContext>, ActionContext> typeToContext :
        identifyingTypeToContext.entrySet()) {
      reporter.handle(
          Event.info(
              String.format(
                  "IdentifyingTypeToContext: \"%s\" = [%s]",
                  typeToContext.getKey(), typeToContext.getValue().getClass().getSimpleName())));
    }
  }

  /**
   * Returns a new {@link Builder} suitable for creating instances of ModuleActionContextRegistry.
   */
  public static Builder builder() {
    return new BuilderImpl();
  }

  /**
   * Builder collecting the contexts and restrictions thereon for a {@link
   * ModuleActionContextRegistry}.
   */
  // TODO(katre): This exists only to allow incremental migration from SpawnActionContextMaps.
  // Delete ASAP.
  public interface Builder {

    /**
     * Restricts the registry to only return implementations for the given type if they were
     * {@linkplain #register registered} with the provided restriction as a command-line identifier.
     *
     * <p>Note that if no registered action context matches the requested command-line identifiers
     * when it is {@linkplain #build() built} then the registry will return {@code null} when
     * queried for this identifying type.
     *
     * <p>This behavior can be reset by passing an empty restriction to this method which will cause
     * the default behavior (last implementation registered for the identifying type) to be used.
     *
     * @param restriction command-line identifier used during registration of the desired
     *     implementation or {@code ""} to allow any implementation of the identifying type
     */
    ModuleActionContextRegistry.Builder restrictTo(Class<?> identifyingType, String restriction);

    /**
     * Registers an action context implementation identified by the given type and which can be
     * {@linkplain #restrictTo restricted} by its provided command-line identifiers.
     */
    <T extends ActionContext> ModuleActionContextRegistry.Builder register(
        Class<T> identifyingType, T context, String... commandLineIdentifiers);

    /** Constructs the registry configured by this builder. */
    ModuleActionContextRegistry build() throws ExecutorInitException;
  }

  /**
   * Builder collecting the contexts and restrictions thereon for a {@link
   * ModuleActionContextRegistry}.
   */
  private static final class BuilderImpl implements Builder {

    private final List<ActionContextInformation<?>> actionContexts = new ArrayList<>();
    private final Map<Class<?>, String> typeToRestriction = new HashMap<>();

    /**
     * Restricts the registry to only return implementations for the given type if they were
     * {@linkplain #register registered} with the provided restriction as a command-line identifier.
     *
     * <p>Note that if no registered action context matches the requested command-line identifiers
     * when it is {@linkplain #build() built} then the registry will return {@code null} when
     * queried for this identifying type.
     *
     * <p>This behavior can be reset by passing an empty restriction to this method which will cause
     * the default behavior (last implementation registered for the identifying type) to be used.
     *
     * @param restriction command-line identifier used during registration of the desired
     *     implementation or {@code ""} to allow any implementation of the identifying type
     */
    @Override
    public Builder restrictTo(Class<?> identifyingType, String restriction) {
      typeToRestriction.put(identifyingType, restriction);
      return this;
    }

    /**
     * Registers an action context implementation identified by the given type and which can be
     * {@linkplain #restrictTo restricted} by its provided command-line identifiers.
     */
    @Override
    public <T extends ActionContext> Builder register(
        Class<T> identifyingType, T context, String... commandLineIdentifiers) {
      actionContexts.add(
          new AutoValue_ModuleActionContextRegistry_ActionContextInformation<>(
              context, identifyingType, ImmutableList.copyOf(commandLineIdentifiers)));
      return this;
    }

    /** Constructs the registry configured by this builder. */
    @Override
    public ModuleActionContextRegistry build() throws ExecutorInitException {
      HashSet<Class<?>> usedTypes = new HashSet<>();
      MutableClassToInstanceMap<ActionContext> contextToInstance =
          MutableClassToInstanceMap.create();
      for (ActionContextInformation<?> actionContextInformation : actionContexts) {
        Class<? extends ActionContext> identifyingType = actionContextInformation.identifyingType();
        if (typeToRestriction.containsKey(identifyingType)) {
          String restriction = typeToRestriction.get(identifyingType);
          if (!actionContextInformation.commandLineIdentifiers().contains(restriction)
              && !restriction.isEmpty()) {
            continue;
          }
        }
        usedTypes.add(identifyingType);
        actionContextInformation.addToMap(contextToInstance);
      }

      Sets.SetView<Class<?>> unusedRestrictions =
          Sets.difference(typeToRestriction.keySet(), usedTypes);
      if (!unusedRestrictions.isEmpty()) {
        throw new ExecutorInitException(
            getMissingIdentifierErrorMessage(unusedRestrictions).toString(),
            ExitCode.COMMAND_LINE_ERROR);
      }

      return new ModuleActionContextRegistry(ImmutableClassToInstanceMap.copyOf(contextToInstance));
    }

    private StringBuilder getMissingIdentifierErrorMessage(
        Sets.SetView<Class<?>> unusedRestrictions) {
      Multimap<Class<?>, String> typeToAvailableIdentifiers = ArrayListMultimap.create();
      for (Class<?> type : unusedRestrictions) {
        for (ActionContextInformation<?> actionContextInformation : actionContexts) {
          if (actionContextInformation.identifyingType().equals(type)) {
            typeToAvailableIdentifiers.putAll(
                type, actionContextInformation.commandLineIdentifiers());
          }
        }
      }
      StringBuilder message = new StringBuilder();
      for (Map.Entry<Class<?>, Collection<String>> typeToIdentifiers :
          typeToAvailableIdentifiers.asMap().entrySet()) {
        Class<?> type = typeToIdentifiers.getKey();
        message.append(
            String.format(
                "No context of type %s registered for requested value '%s', available identifiers"
                    + " are: [%s]%n",
                type.getSimpleName(),
                typeToRestriction.get(type),
                Joiner.on(", ").join(typeToIdentifiers.getValue())));
      }
      return message;
    }
  }

  @AutoValue
  abstract static class ActionContextInformation<T extends ActionContext> {

    abstract T context();

    abstract Class<T> identifyingType();

    abstract ImmutableList<String> commandLineIdentifiers();

    private void addToMap(MutableClassToInstanceMap<ActionContext> map) {
      map.putInstance(identifyingType(), context());
    }
  }
}
