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
package com.google.devtools.build.lib.rules;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import javax.annotation.Nullable;

/**
 * This configured target pretends to be whatever type of target "actual" is, returning its label,
 * transitive info providers and target.
 *
 * <p>Transitive info providers can also be overridden.
 */
@AutoCodec
@Immutable
public final class AliasConfiguredTarget implements ConfiguredTarget, ClassObject {
  private final Label label;
  private final BuildConfigurationValue.Key configurationKey;
  private final ConfiguredTarget actual;
  private final ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider>
      overrides;
  private final ImmutableMap<Label, ConfigMatchingProvider> configConditions;

  public AliasConfiguredTarget(
      RuleContext ruleContext,
      ConfiguredTarget actual,
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> overrides) {
    this(
        ruleContext.getLabel(),
        Preconditions.checkNotNull(ruleContext.getConfigurationKey()),
        Preconditions.checkNotNull(actual),
        Preconditions.checkNotNull(overrides),
        ruleContext.getConfigConditions());
  }

  @AutoCodec.Instantiator
  @VisibleForSerialization
  AliasConfiguredTarget(
      Label label,
      BuildConfigurationValue.Key configurationKey,
      ConfiguredTarget actual,
      ImmutableMap<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> overrides,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions) {
    this.label = label;
    this.configurationKey = configurationKey;
    this.actual = actual;
    this.overrides = overrides;
    this.configConditions = configConditions;
  }

  public ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions() {
    return configConditions;
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    if (overrides.containsKey(provider)) {
      return provider.cast(overrides.get(provider));
    }

    return actual.getProvider(provider);
  }

  @Override
  public Label getLabel() {
    return actual.getLabel();
  }

  @Override
  public Object get(String providerKey) {
    return actual.get(providerKey);
  }

  @Nullable
  @Override
  public Info get(Provider.Key providerKey) {
    return actual.get(providerKey);
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    return actual.getIndex(semantics, key);
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    return actual.containsKey(semantics, key);
  }

  @Override
  public BuildConfigurationValue.Key getConfigurationKey() {
    // This does not return actual.getConfigurationKey() because actual might be an input file, in
    // which case its configurationKey is null and we don't want to have rules that have a null
    // configurationKey.
    return configurationKey;
  }

  /* ClassObject methods */

  @Override
  public Object getValue(String name) {
    if (name.equals(LABEL_FIELD)) {
      return getLabel();
    } else if (name.equals(FILES_FIELD)) {
      // A shortcut for files to build in Starlark. FileConfiguredTarget and RuleConfiguredTarget
      // always has FileProvider and Error- and PackageGroupConfiguredTarget-s shouldn't be
      // accessible in Starlark.
      return Depset.of(Artifact.TYPE, getProvider(FileProvider.class).getFilesToBuild());
    }
    return actual.getValue(name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return actual.getFieldNames();
  }

  @Override
  public String getErrorMessageForUnknownField(String name) {
    // Use the default error message.
    return null;
  }

  @Override
  public ConfiguredTarget getActual() {
    // This will either dereference an alias chain, or return the final ConfiguredTarget.
    return actual.getActual();
  }

  @Override
  public Label getOriginalLabel() {
    return label;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<alias target " + label + " of " + actual.getLabel() + ">");
  }
}
