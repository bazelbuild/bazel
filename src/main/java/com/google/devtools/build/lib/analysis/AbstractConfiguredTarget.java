// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Printer;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * An abstract implementation of ConfiguredTarget in which all properties are
 * assigned trivial default values.
 */
public abstract class AbstractConfiguredTarget
    implements ConfiguredTarget, VisibilityProvider, ClassObject {
  private final Target target;
  private final BuildConfiguration configuration;

  private final NestedSet<PackageSpecification> visibility;

  // Cached on-demand default provider
  private final AtomicReference<DefaultProvider> defaultProvider = new AtomicReference<>();

  // Accessors for Skylark
  private static final String DATA_RUNFILES_FIELD = "data_runfiles";
  private static final String DEFAULT_RUNFILES_FIELD = "default_runfiles";

  AbstractConfiguredTarget(Target target,
                           BuildConfiguration configuration) {
    this.target = target;
    this.configuration = configuration;
    this.visibility = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  AbstractConfiguredTarget(TargetContext targetContext) {
    this.target = targetContext.getTarget();
    this.configuration = targetContext.getConfiguration();
    this.visibility = targetContext.getVisibility();
  }

  @Override
  public final NestedSet<PackageSpecification> getVisibility() {
    return visibility;
  }

  @Override
  public Target getTarget() {
    return target;
  }

  @Override
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  @Override
  public Label getLabel() {
    return getTarget().getLabel();
  }

  @Override
  public String toString() {
    return "ConfiguredTarget(" + getTarget().getLabel() + ", " + getConfiguration() + ")";
  }

  @Override
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
    AnalysisUtils.checkProvider(provider);
    if (provider.isAssignableFrom(getClass())) {
      return provider.cast(this);
    } else {
      return null;
    }
  }

  @Override
  public Object getValue(String name) {
    switch (name) {
      case LABEL_FIELD:
        return getLabel();
      default:
        return get(name);
    }
  }

  @Override
  public final Object getIndex(Object key, Location loc) throws EvalException {
    if (!(key instanceof Provider)) {
      throw new EvalException(loc, String.format(
          "Type Target only supports indexing by object constructors, got %s instead",
          EvalUtils.getDataTypeName(key)));
    }
    Provider constructor = (Provider) key;
    Object declaredProvider = get(constructor.getKey());
    if (declaredProvider != null) {
      return declaredProvider;
    }
    throw new EvalException(loc, Printer.format(
        "%r%s doesn't contain declared provider '%s'",
        this,
        getTarget().getAssociatedRule() == null ? ""
            : " (rule '" + getTarget().getAssociatedRule().getRuleClass() + "')",
        constructor.getPrintableName()));
  }

  @Override
  public boolean containsKey(Object key, Location loc) throws EvalException {
    if (!(key instanceof Provider)) {
      throw new EvalException(loc, String.format(
          "Type Target only supports querying by object constructors, got %s instead",
          EvalUtils.getDataTypeName(key)));
    }
    return get(((Provider) key).getKey()) != null;
  }

  @Override
  public String errorMessage(String name) {
    return null;
  }

  @Override
  public final ImmutableCollection<String> getKeys() {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    result.addAll(ImmutableList.of(
        DATA_RUNFILES_FIELD,
        DEFAULT_RUNFILES_FIELD,
        LABEL_FIELD,
        FILES_FIELD,
        FilesToRunProvider.SKYLARK_NAME));
    if (get(OutputGroupProvider.SKYLARK_CONSTRUCTOR) != null) {
      result.add(OutputGroupProvider.SKYLARK_NAME);
    }
    addExtraSkylarkKeys(result::add);
    return result.build();
  }

  protected void addExtraSkylarkKeys(Consumer<String> result) {
  }

  private DefaultProvider getDefaultProvider() {
    if (defaultProvider.get() == null) {
      defaultProvider.compareAndSet(
          null,
          DefaultProvider.build(
              getProvider(RunfilesProvider.class),
              getProvider(FileProvider.class),
              getProvider(FilesToRunProvider.class)));
    }
    return defaultProvider.get();
  }

  /** Returns a declared provider provided by this target. Only meant to use from Skylark. */
  @Nullable
  @Override
  public final Info get(Provider.Key providerKey) {
    if (providerKey.equals(DefaultProvider.SKYLARK_CONSTRUCTOR.getKey())) {
      return getDefaultProvider();
    }
    return rawGetSkylarkProvider(providerKey);
  }

  /** Implement in subclasses to get a skylark provider for a given {@code providerKey}. */
  @Nullable
  protected abstract Info rawGetSkylarkProvider(Provider.Key providerKey);

  /**
   * Returns a value provided by this target. Only meant to use from Skylark.
   */
  @Override
  public final Object get(String providerKey) {
    if (OutputGroupProvider.SKYLARK_NAME.equals(providerKey)) {
      return get(OutputGroupProvider.SKYLARK_CONSTRUCTOR);
    }
    switch (providerKey) {
      case FILES_FIELD:
      case DEFAULT_RUNFILES_FIELD:
      case DATA_RUNFILES_FIELD:
      case FilesToRunProvider.SKYLARK_NAME:
        // Standard fields should be proxied to their default provider object
        return getDefaultProvider().getValue(providerKey);
      case OutputGroupProvider.SKYLARK_NAME:
        return get(OutputGroupProvider.SKYLARK_CONSTRUCTOR);
      default:
        return rawGetSkylarkProvider(providerKey);
    }
  }

  /** Implement in subclasses to get a skylark provider for a given {@code providerKey}. */
  protected abstract Object rawGetSkylarkProvider(String providerKey);

  // All main target classes must override this method to provide more descriptive strings.
  // Exceptions are currently EnvironmentGroupConfiguredTarget and PackageGroupConfiguredTarget.
  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<unknown target " + getTarget().getLabel() + ">");
  }
}
