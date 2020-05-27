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

package com.google.devtools.build.lib.analysis.configuredtargets;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.VisibilityProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * An abstract implementation of ConfiguredTarget in which all properties are assigned trivial
 * default values.
 */
public abstract class AbstractConfiguredTarget implements ConfiguredTarget, VisibilityProvider {
  private final Label label;
  private final BuildConfigurationValue.Key configurationKey;

  private final NestedSet<PackageGroupContents> visibility;

  // Cached on-demand default provider
  private final AtomicReference<DefaultInfo> defaultProvider = new AtomicReference<>();

  // Accessors for Starlark
  private static final String DATA_RUNFILES_FIELD = "data_runfiles";
  private static final String DEFAULT_RUNFILES_FIELD = "default_runfiles";

  /**
   * The name of the key for the 'actions' synthesized provider.
   *
   * <p>If you respond to this key you are expected to return a list of actions belonging to this
   * configured target.
   */
  public static final String ACTIONS_FIELD_NAME = "actions";

  // A set containing all field names which may be specially handled (and thus may not be
  // attributed to normal user-specified providers).
  private static final ImmutableSet<String> SPECIAL_FIELD_NAMES =
      ImmutableSet.of(
          LABEL_FIELD,
          FILES_FIELD,
          DEFAULT_RUNFILES_FIELD,
          DATA_RUNFILES_FIELD,
          FilesToRunProvider.STARLARK_NAME,
          OutputGroupInfo.STARLARK_NAME,
          ACTIONS_FIELD_NAME);

  public AbstractConfiguredTarget(Label label, BuildConfigurationValue.Key configurationKey) {
    this(label, configurationKey, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  protected AbstractConfiguredTarget(
      Label label,
      BuildConfigurationValue.Key configurationKey,
      NestedSet<PackageGroupContents> visibility) {
    this.label = label;
    this.configurationKey = configurationKey;
    this.visibility = visibility;
  }

  @Override
  public boolean isImmutable() {
    return true; // all Targets are immutable and Starlark-hashable
  }

  @Override
  public final NestedSet<PackageGroupContents> getVisibility() {
    return visibility;
  }

  @Override
  public BuildConfigurationValue.Key getConfigurationKey() {
    return configurationKey;
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public String toString() {
    return "ConfiguredTarget(" + getLabel() + ", " + getConfigurationChecksum() + ")";
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
  public Object getValue(StarlarkSemantics semantics, String name) throws EvalException {
    if (semantics.incompatibleDisableTargetProviderFields()
        && !SPECIAL_FIELD_NAMES.contains(name)) {
      throw Starlark.errorf(
          "Accessing providers via the field syntax on structs is "
              + "deprecated and will be removed soon. It may be temporarily re-enabled by setting "
              + "--incompatible_disable_target_provider_fields=false. See "
              + "https://github.com/bazelbuild/bazel/issues/9014 for details.");
    }
    return getValue(name);
  }

  @Override
  public Object getValue(String name) {
    switch (name) {
      case LABEL_FIELD:
        return getLabel();
      case ACTIONS_FIELD_NAME:
        // Depending on subclass, the 'actions' field will either be unsupported or of type
        // java.util.List, which needs to be converted to Sequence before being returned.
        Object result = get(name);
        return result != null ? Starlark.fromJava(result, null) : null;
      default:
        return get(name);
    }
  }

  @Override
  public final Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    if (!(key instanceof Provider)) {
      throw Starlark.errorf(
          "Type Target only supports indexing by object constructors, got %s instead",
          Starlark.type(key));
    }
    Provider constructor = (Provider) key;
    Object declaredProvider = get(constructor.getKey());
    if (declaredProvider != null) {
      return declaredProvider;
    }
    throw Starlark.errorf(
        "%s%s doesn't contain declared provider '%s'",
        Starlark.repr(this),
        getRuleClassString().isEmpty() ? "" : " (rule '" + getRuleClassString() + "')",
        constructor.getPrintableName());
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    if (!(key instanceof Provider)) {
      throw Starlark.errorf(
          "Type Target only supports querying by object constructors, got %s instead",
          Starlark.type(key));
    }
    return get(((Provider) key).getKey()) != null;
  }

  @Override
  public String getErrorMessageForUnknownField(String name) {
    return null;
  }

  @Override
  public final ImmutableCollection<String> getFieldNames() {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    result.addAll(
        ImmutableList.of(
            DATA_RUNFILES_FIELD,
            DEFAULT_RUNFILES_FIELD,
            LABEL_FIELD,
            FILES_FIELD,
            FilesToRunProvider.STARLARK_NAME));
    if (get(OutputGroupInfo.STARLARK_CONSTRUCTOR) != null) {
      result.add(OutputGroupInfo.STARLARK_NAME);
    }
    addExtraStarlarkKeys(result::add);
    return result.build();
  }

  protected void addExtraStarlarkKeys(Consumer<String> result) {}

  private DefaultInfo getDefaultProvider() {
    if (defaultProvider.get() == null) {
      defaultProvider.compareAndSet(
          null,
          DefaultInfo.build(
              getProvider(RunfilesProvider.class),
              getProvider(FileProvider.class),
              getProvider(FilesToRunProvider.class)));
    }
    return defaultProvider.get();
  }

  /** Returns a declared provider provided by this target. Only meant to use from Starlark. */
  @Nullable
  @Override
  public final Info get(Provider.Key providerKey) {
    if (providerKey.equals(DefaultInfo.PROVIDER.getKey())) {
      return getDefaultProvider();
    }
    return rawGetStarlarkProvider(providerKey);
  }

  /** Implement in subclasses to get a Starlark provider for a given {@code providerKey}. */
  @Nullable
  protected abstract Info rawGetStarlarkProvider(Provider.Key providerKey);

  public String getRuleClassString() {
    return "";
  }

  /** Returns a value provided by this target. Only meant to use from Starlark. */
  @Override
  public final Object get(String providerKey) {
    switch (providerKey) {
      case FILES_FIELD:
        return getDefaultProvider().getFiles();
      case DEFAULT_RUNFILES_FIELD:
        return getDefaultProvider().getDefaultRunfiles();
      case DATA_RUNFILES_FIELD:
        return getDefaultProvider().getDataRunfiles();
      case FilesToRunProvider.STARLARK_NAME:
        return getDefaultProvider().getFilesToRun();
      case OutputGroupInfo.STARLARK_NAME:
        return get(OutputGroupInfo.STARLARK_CONSTRUCTOR);
      default:
        return rawGetStarlarkProvider(providerKey);
    }
  }

  /** Implement in subclasses to get a Starlark provider for a given {@code providerKey}. */
  protected abstract Object rawGetStarlarkProvider(String providerKey);

  // All main target classes must override this method to provide more descriptive strings.
  // Exceptions are currently EnvironmentGroupConfiguredTarget and PackageGroupConfiguredTarget.
  @Override
  public void repr(Printer printer) {
    printer.append("<unknown target " + getLabel() + ">");
  }
}
