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

import com.google.auto.value.AutoValue;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlag;
import java.util.Map;

/**
 * Provider for checking the set of feature flags used by an android_binary.
 *
 * <p>Because the feature flags are completely replaced by android_binary, android_test uses this
 * provider to ensure that the test sets the same flags as the binary. Otherwise, the dependencies
 * of the android_test will be compiled with different flags from the android_binary code which runs
 * in the same Android virtual machine, which may cause compatibility issues at runtime.
 */
@AutoValue
@Immutable
public abstract class AndroidFeatureFlagSetProvider implements TransitiveInfoProvider {

  /** The name of the attribute used by Android rules to set config_feature_flags. */
  public static final String FEATURE_FLAG_ATTR = "feature_flags";

  AndroidFeatureFlagSetProvider() {}

  /** Creates a new AndroidFeatureFlagSetProvider with the given flags. */
  public static AndroidFeatureFlagSetProvider create(Optional<? extends Map<Label, String>> flags) {
    return new AutoValue_AndroidFeatureFlagSetProvider(flags.transform(ImmutableMap::copyOf));
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
      ruleContext.attributeError(
          FEATURE_FLAG_ATTR,
          String.format(
              "the %s attribute is not available in package '%s'",
              FEATURE_FLAG_ATTR,
              ruleContext.getLabel().getPackageIdentifier()));
      throw new RuleErrorException();
    }

    Iterable<? extends TransitiveInfoCollection> actualTargets =
        ruleContext.getPrerequisites(FEATURE_FLAG_ATTR, Mode.TARGET);
    boolean aliasFound = false;
    for (TransitiveInfoCollection target : actualTargets) {
      Label label = AliasProvider.getDependencyLabel(target);
      if (!label.equals(target.getLabel())) {
        ruleContext.attributeError(
            FEATURE_FLAG_ATTR,
            String.format(
                "Feature flags must be named directly, not through aliases; use '%s', not '%s'",
                target.getLabel(), label));
        aliasFound = true;
      }
    }
    if (aliasFound) {
      throw new RuleErrorException();
    }
    return Optional.of(ImmutableMap.copyOf(expectedValues));
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

  public abstract Optional<ImmutableMap<Label, String>> getFlags();
}
