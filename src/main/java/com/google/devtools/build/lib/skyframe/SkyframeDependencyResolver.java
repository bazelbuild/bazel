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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Function;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.DependencyResolver;
import com.google.devtools.build.lib.view.TargetAndConfiguration;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A dependency resolver for use within Skyframe. Loads packages lazily when possible.
 */
public final class SkyframeDependencyResolver extends DependencyResolver {
  private static final Function<TargetAndConfiguration, SkyKey> TO_KEYS =
      new Function<TargetAndConfiguration, SkyKey>() {
        @Override
        public SkyKey apply(TargetAndConfiguration input) {
          Label depLabel = input.getLabel();
          return ConfiguredTargetValue.key(depLabel, input.getConfiguration());
        }
      };

  /**
   * Exception thrown by {@link DependencyResolver}.
   */
  public static class DependencyResolverException extends SkyFunctionException {
    public DependencyResolverException(SkyKey key, NoSuchTargetException e) {
      super(key, e, Transience.PERSISTENT);
    }

    public DependencyResolverException(SkyKey key, ConfiguredValueCreationException e) {
      super(key, e, Transience.PERSISTENT);
    }

    public DependencyResolverException(SkyKey key, EvalException e) {
      super(key, e, Transience.PERSISTENT);
    }
  }

  /**
   * An exception indicating that there was a problem during the construction of
   * a ConfiguredTargetValue.
   */
  public static final class ConfiguredValueCreationException extends Exception {

    public ConfiguredValueCreationException(String message) {
      super(message);
    }
  }

  private final Environment env;

  public SkyframeDependencyResolver(Environment env) {
    this.env = env;
  }

  @Override
  protected void invalidVisibilityReferenceHook(TargetAndConfiguration value, Label label) {
    env.getListener().handle(
        Event.error(TargetUtils.getLocationMaybe(value.getTarget()), String.format(
            "Label '%s' in visibility attribute does not refer to a package group", label)));
  }

  @Override
  protected void invalidPackageGroupReferenceHook(TargetAndConfiguration value, Label label) {
    env.getListener().handle(
        Event.error(TargetUtils.getLocationMaybe(value.getTarget()), String.format(
            "label '%s' does not refer to a package group", label)));
  }

  @Nullable
  @Override
  protected Target getTarget(Label label) throws NoSuchThingException {
    if (env.getValue(TargetMarkerValue.key(label)) == null) {
      return null;
    }
    SkyKey key = PackageValue.key(label.getPackageIdentifier());
    SkyValue value = env.getValue(key);
    if (value == null) {
      return null;
    }
    PackageValue packageValue = (PackageValue) value;
    return packageValue.getPackage().getTarget(label.getName());
  }

  /**
   * Returns the set of {@link com.google.devtools.build.lib.view.config.ConfigMatchingProvider}s
   * that key the configurable attributes used by this rule.
   *
   * <p>>If the configured targets supplying those providers aren't yet resolved by the
   * dependency resolver, returns null.
   */
  public Set<ConfigMatchingProvider> getConfigConditions(Target target, Environment env,
      DependencyResolver resolver, TargetAndConfiguration ctgValue, SkyKey skyKey)
      throws DependencyResolverException {
    if (!(target instanceof Rule)) {
      return ImmutableSet.of();
    }

    ImmutableSet.Builder<ConfigMatchingProvider> configConditions = ImmutableSet.builder();

    // Collect the labels of the configured targets we need to resolve.
    ListMultimap<Attribute, Label> configLabelMap = ArrayListMultimap.create();
    RawAttributeMapper attributeMap = RawAttributeMapper.of(((Rule) target));
    for (Attribute a : ((Rule) target).getAttributes()) {
      for (Label configLabel : attributeMap.getConfigurabilityKeys(a.getName(), a.getType())) {
        if (!Type.Selector.isReservedLabel(configLabel)) {
          configLabelMap.put(a, configLabel);
        }
      }
    }
    if (configLabelMap.isEmpty()) {
      return ImmutableSet.of();
    }

    // Collect the corresponding Skyframe configured target values. Abort early if they haven't
    // been computed yet.
    ListMultimap<Attribute, TargetAndConfiguration> configValueMap =
        resolver.resolveRuleLabels(ctgValue, configLabelMap);
    Map<SkyKey, ConfiguredTargetValue> configValues =
        resolveDependencies(env, configValueMap, skyKey, target);
    if (configValues == null) {
      return null;
    }

    // Get the configured targets as ConfigMatchingProvider interfaces.
    for (TargetAndConfiguration entry : configValueMap.values()) {
      ConfiguredTargetValue value = configValues.get(TO_KEYS.apply(entry));
      // The code above guarantees that value is non-null here.
      ConfigMatchingProvider provider =
          value.getConfiguredTarget().getProvider(ConfigMatchingProvider.class);
      if (provider != null) {
        configConditions.add(provider);
      } else {
        // Not a valid provider for configuration conditions.
        Target badTarget = entry.getTarget();
        String message = badTarget + " is not a valid configuration key for "
            + target.getLabel().toString();
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(badTarget), message));
        throw new DependencyResolverException(skyKey,
            new ConfiguredValueCreationException(message));
      }
    }

    return configConditions.build();
  }

  /***
   * Resolves the targets referenced in depValueNames and returns their ConfiguredTarget
   * instances.
   *
   * <p>Returns null if not all instances are available yet.
   *
   */
  public Map<SkyKey, ConfiguredTargetValue> resolveDependencies(Environment env,
      ListMultimap<Attribute, TargetAndConfiguration> deps, SkyKey skyKey,
      Target target) throws DependencyResolverException {
    boolean ok = !env.valuesMissing();
    String message = null;
    Iterable<SkyKey> depKeys = Iterables.transform(deps.values(), TO_KEYS);
    // TODO(bazel-team): maybe having a two-exception argument is better than typing a generic
    // Exception here.
    Map<SkyKey, ValueOrException2<NoSuchTargetException,
        NoSuchPackageException>> depValuesOrExceptions = env.getValuesOrThrow(depKeys,
        NoSuchTargetException.class, NoSuchPackageException.class);
    Map<SkyKey, ConfiguredTargetValue> depValues = new HashMap<>(depValuesOrExceptions.size());
    for (Map.Entry<SkyKey, ValueOrException2<NoSuchTargetException, NoSuchPackageException>> entry
        : depValuesOrExceptions.entrySet()) {
      LabelAndConfiguration depLabelAndConfiguration =
          (LabelAndConfiguration) entry.getKey().argument();
      Label depLabel = depLabelAndConfiguration.getLabel();
      ConfiguredTargetValue depValue = null;
      NoSuchThingException directChildException = null;
      try {
        depValue = (ConfiguredTargetValue) entry.getValue().get();
      } catch (NoSuchTargetException e) {
        if (depLabel.equals(e.getLabel())) {
          directChildException = e;
        }
      } catch (NoSuchPackageException e) {
        if (depLabel.getPackageName().equals(e.getPackageName())) {
          directChildException = e;
        }
      }
      // If an exception wasn't caused by a direct child target value, we'll treat it the same
      // as any other missing dep by setting ok = false below, and returning null at the end.
      if (directChildException != null) {
        // Only update messages for missing targets we depend on directly.
        message = TargetUtils.formatMissingEdge(target, depLabel, directChildException);
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(target), message));
      }

      if (depValue == null) {
        ok = false;
      } else {
        depValues.put(entry.getKey(), depValue);
      }
    }
    if (message != null) {
      throw new DependencyResolverException(skyKey, new NoSuchTargetException(message));
    }
    if (!ok) {
      return null;
    } else {
      return depValues;
    }
  }
}
