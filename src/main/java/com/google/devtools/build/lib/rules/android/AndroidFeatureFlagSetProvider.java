// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlag;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidFeatureFlagSetProviderApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import java.util.Map;

/**
 * Provider for checking the set of feature flags used by an android_binary.
 *
 * <p>Because the feature flags are completely replaced by android_binary, android_test uses this
 * provider to ensure that the test sets the same flags as the binary. Otherwise, the dependencies
 * of the android_test will be compiled with different flags from the android_binary code which runs
 * in the same Android virtual machine, which may cause compatibility issues at runtime.
 */
@Immutable
public final class AndroidFeatureFlagSetProvider extends NativeInfo
    implements AndroidFeatureFlagSetProviderApi {

  public static final Provider PROVIDER = new Provider();

  /** The name of the attribute used by Android rules to set config_feature_flags. */
  public static final String FEATURE_FLAG_ATTR = "feature_flags";

  private final Optional<ImmutableMap<Label, String>> flags;

  AndroidFeatureFlagSetProvider(Optional<? extends Map<Label, String>> flags) {
    super(PROVIDER);
    this.flags = flags.transform(ImmutableMap::copyOf);
  }

  public static AndroidFeatureFlagSetProvider create(Optional<? extends Map<Label, String>> flags) {
    return new AndroidFeatureFlagSetProvider(flags.transform(ImmutableMap::copyOf));
  }

  /**
   * Constructs a definition for the attribute used to restrict access to feature flags. The
   * allowlist will only be reached if the feature_flags attribute is explicitly set.
   */
  public static Attribute.Builder<Label> getAllowlistAttribute(RuleDefinitionEnvironment env) {
    return ConfigFeatureFlag.getAllowlistAttribute(env, FEATURE_FLAG_ATTR);
  }

  /**
   * Builds a map which can be used with create, confirming that the desired flag values were
   * actually received, and producing an error if they were not (because aliases were used).
   *
   * <p>If the attribute which defines feature flags was not specified, an empty {@link Optional}
   * instance is returned.
   */
  public static Optional<ImmutableMap<Label, String>> getAndValidateFlagMapFromRuleContext(
      RuleContext ruleContext) throws RuleErrorException {
    NonconfigurableAttributeMapper attrs = NonconfigurableAttributeMapper.of(ruleContext.getRule());

    if (!attrs.isAttributeValueExplicitlySpecified(FEATURE_FLAG_ATTR)) {
      return Optional.absent();
    }

    Map<Label, String> expectedValues =
        attrs.get(FEATURE_FLAG_ATTR, BuildType.LABEL_KEYED_STRING_DICT);
    if (expectedValues.isEmpty()) {
      return Optional.of(ImmutableMap.of());
    }

    if (!ConfigFeatureFlag.isAvailable(ruleContext)) {
      throw ruleContext.throwWithAttributeError(
          FEATURE_FLAG_ATTR,
          String.format(
              "the %s attribute is not available in package '%s'",
              FEATURE_FLAG_ATTR, ruleContext.getLabel().getPackageIdentifier()));
    }

    Iterable<? extends TransitiveInfoCollection> actualTargets =
        ruleContext.getPrerequisites(FEATURE_FLAG_ATTR, TransitionMode.TARGET);
    RuleErrorException exception = null;
    for (TransitiveInfoCollection target : actualTargets) {
      Label label = AliasProvider.getDependencyLabel(target);
      if (!label.equals(target.getLabel())) {
        try {
          exception =
              ruleContext.throwWithAttributeError(
                  FEATURE_FLAG_ATTR,
                  String.format(
                      "Feature flags must be named directly, not through aliases; use '%s', not"
                          + " '%s'",
                      target.getLabel(), label));
        } catch (RuleErrorException e) {
          exception = e;
        }
      }
    }
    if (exception != null) {
      throw exception;
    }
    return Optional.of(ImmutableMap.copyOf(expectedValues));
  }

  /** Returns the feature flags this rule sets as user-friendly strings. */
  public static ImmutableSet<String> getFlagNames(RuleContext ruleContext) {
    return ruleContext
        .attributes()
        .get(AndroidFeatureFlagSetProvider.FEATURE_FLAG_ATTR, BuildType.LABEL_KEYED_STRING_DICT)
        .keySet()
        .stream()
        .map(label -> label.toString())
        .collect(ImmutableSet.toImmutableSet());
  }

  /**
   * Returns whether it is acceptable to have a dependency with flags {@code depFlags} if the target
   * has flags {@code targetFlags}.
   */
  public static boolean isValidDependency(
      Optional<? extends Map<Label, String>> targetFlags,
      Optional<? extends Map<Label, String>> depFlags) {
    return !depFlags.isPresent()
        || (targetFlags.isPresent() && targetFlags.get().equals(depFlags.get()));
  }

  public Optional<ImmutableMap<Label, String>> getFlags() {
    return flags;
  }

  @Override
  public ImmutableMap<Label, String> getFlagMap() {
    return flags.or(ImmutableMap.of());
  }

  /** Provider class for {@link AndroidFeatureFlagSetProvider} objects. */
  public static class Provider extends BuiltinProvider<AndroidFeatureFlagSetProvider>
      implements AndroidFeatureFlagSetProviderApi.Provider {
    private Provider() {
      super(NAME, AndroidFeatureFlagSetProvider.class);
    }

    public String getName() {
      return NAME;
    }

    @Override
    public AndroidFeatureFlagSetProvider create(Dict<?, ?> flags) // <Label, String>
        throws EvalException {
      return new AndroidFeatureFlagSetProvider(
          Optional.of(Dict.noneableCast(flags, Label.class, String.class, "flags")));
    }
  }
}
