// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.devtools.build.lib.server.FailureDetails.TargetPatterns.Code.DEPENDENCY_NOT_FOUND;
import static com.google.devtools.common.options.OptionsParser.STARLARK_SKIPPED_PREFIXES;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.PackageContext;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.runtime.StarlarkOptionsParser;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Converts a list of command-line flags (like {@code --compilation_mode=dbg} or {@code
 * --//custom/starlark:flag=foo}) into a {@link NativeAndStarlarkFlags} instance. This is intended
 * as preparation for using the flags to create or update a build configuration in Bazel.
 */
public final class ParsedFlagsFunction implements SkyFunction {
  private final ImmutableSet<Class<? extends FragmentOptions>> optionsClasses;

  public ParsedFlagsFunction(ImmutableSet<Class<? extends FragmentOptions>> optionsClasses) {
    this.optionsClasses = optionsClasses;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, SkyFunctionException {
    ParsedFlagsValue.Key key = (ParsedFlagsValue.Key) skyKey.argument();

    ImmutableList.Builder<String> nativeFlags = ImmutableList.builder();
    ImmutableList.Builder<String> starlarkFlags = ImmutableList.builder();
    ImmutableMap<String, Label> flagAliasMappings = key.flagAliasMappings();
    for (String flagSetting : key.rawFlags()) {
      if (!flagSetting.startsWith("--")) {
        // This is either something like "-c" or an invalid setting. Let options parsing handle it.
        nativeFlags.add(flagSetting);
        continue;
      }
      String flagName;
      String flagValue = "";
      int delimiterIndex = flagSetting.indexOf("=");
      boolean noPrefix = false;
      if (delimiterIndex != -1) {
        flagName = flagSetting.substring(2, delimiterIndex); // --flag=value
        flagValue = flagSetting.substring(delimiterIndex + 1);
      } else if (flagSetting.startsWith("--no")) {
        flagName = flagSetting.substring(4); // --no<flag>
        noPrefix = true;
      } else {
        flagName = flagSetting.substring(2); // --<flag>
      }
      // If --flag_alias=foo=//bar and we see --foo=1, use the canonical setting --//bar=1.
      Label actualFlag = flagAliasMappings.get(flagName);
      if (actualFlag != null) {
        flagSetting =
            "--%s%s%s"
                .formatted(
                    noPrefix ? "no" : "",
                    actualFlag.getUnambiguousCanonicalForm(),
                    delimiterIndex == -1 ? "" : "=" + flagValue);
      }
      if (STARLARK_SKIPPED_PREFIXES.stream().noneMatch(flagSetting::startsWith)) {
        nativeFlags.add(flagSetting);
      } else {
        starlarkFlags.add(flagSetting);
      }
    }
    // The StarlarkOptionsParser needs a native options parser to handle some forms of value
    // conversion and as a place to inject the flag values.
    // TODO: https://github.com/bazelbuild/bazel/issues/22365 - Clean this up as part of a general
    // rewrite.
    OptionsParser fakeNativeParser =
        OptionsParser.builder().withConversionContext(key.packageContext()).build();
    StarlarkOptionsParser starlarkFlagParser =
        StarlarkOptionsParser.builder()
            .buildSettingLoader(new SkyframeTargetLoader(env, key.packageContext()))
            .nativeOptionsParser(fakeNativeParser)
            .includeDefaultValues(key.includeDefaultValues())
            .allowNonFlagBuildSettings(key.allowNonFlagBuildSettings())
            .build();
    try {
      if (!starlarkFlagParser.parseGivenArgs(starlarkFlags.build())) {
        return null;
      }
    } catch (OptionsParsingException e) {
      throw new ParsedFlagsFunctionException(e);
    }
    if (key.allowNonFlagBuildSettings()) {
      PackageIdentifier consumerPkg = key.packageContext().packageIdentifier();
      for (Target settingTarget : starlarkFlagParser.getParsedBuildSettingTargets().values()) {
        Rule associatedRule = settingTarget.getAssociatedRule();
        if (associatedRule == null) {
          continue;
        }
        BuildSetting setting = associatedRule.getRuleClassObject().getBuildSetting();
        if (setting != null && !setting.isFlag()) {
          Boolean isVisible = checkPackageVisibility(env, settingTarget, consumerPkg);
          if (isVisible == null) {
            return null;
          }
          if (!isVisible) {
            throw new ParsedFlagsFunctionException(
                new OptionsParsingException(
                    String.format(
                        "Build setting '%s' cannot be set by platform in package '%s': target is"
                            + " not visible from this package",
                        settingTarget.getLabel(), consumerPkg)));
          }
        }
      }
    }
    NativeAndStarlarkFlags.Builder flags =
        NativeAndStarlarkFlags.builder()
            .nativeFlags(nativeFlags.build())
            .starlarkFlags(starlarkFlagParser.getStarlarkOptions())
            .starlarkOptionAllowingMultiple(starlarkFlagParser.getStarlarkOptionsAllowingMultiple())
            .scopesAttributes(starlarkFlagParser.getScopesAttributes())
            .optionsClasses(optionsClasses)
            .repoMapping(key.packageContext().repoMapping());

    if (key.includeDefaultValues()) {
      flags.starlarkFlagDefaults(starlarkFlagParser.getDefaultValues());
    }

    try {
      return ParsedFlagsValue.parseAndCreate(flags.build());
    } catch (OptionsParsingException e) {
      throw new ParsedFlagsFunctionException(e);
    }
  }

  @Nullable
  private static Boolean checkPackageVisibility(
      Environment env, Target settingTarget, PackageIdentifier consumerPkg)
      throws InterruptedException {
    // Canonical visibility checking requires ConfiguredTarget, which doesn't exist yet
    // during flag parsing before configuration creation. We resolve PackageGroups
    // directly via Skyframe PackageValues instead.
    if (settingTarget.getLabel().getPackageIdentifier().equals(consumerPkg)) {
      return true;
    }
    RuleVisibility ruleVisibility =
        settingTarget.isCreatedInSymbolicMacro()
            ? settingTarget.getActualVisibility()
            : settingTarget.getVisibility();
    if (ruleVisibility.equals(RuleVisibility.PUBLIC)) {
      return true;
    }
    if (ruleVisibility.equals(RuleVisibility.PRIVATE)) {
      return false;
    }
    if (!(ruleVisibility instanceof PackageGroupsRuleVisibility packageGroupsVisibility)) {
      return false;
    }
    if (packageGroupsVisibility.getDirectPackages().containsPackage(consumerPkg)) {
      return true;
    }
    ImmutableList<Label> initialGroups = packageGroupsVisibility.getPackageGroups();
    if (initialGroups.isEmpty()) {
      return false;
    }

    Set<Label> visitedGroups = new HashSet<>();
    Set<Label> currentGroups = new LinkedHashSet<>(initialGroups);

    while (!currentGroups.isEmpty()) {
      Set<PackageIdentifier> packageKeys = new HashSet<>();
      for (Label groupLabel : currentGroups) {
        packageKeys.add(groupLabel.getPackageIdentifier());
      }
      visitedGroups.addAll(currentGroups);

      SkyframeLookupResult lookupResult = env.getValuesAndExceptions(packageKeys);
      if (env.valuesMissing()) {
        return null;
      }
      Set<Label> nextGroups = new LinkedHashSet<>();
      for (Label groupLabel : currentGroups) {
        PackageIdentifier pkgId = groupLabel.getPackageIdentifier();
        PackageValue pkgValue;
        try {
          pkgValue = (PackageValue) lookupResult.getOrThrow(pkgId, NoSuchPackageException.class);
        } catch (NoSuchPackageException e) {
          continue;
        }
        if (pkgValue == null) {
          return null;
        }
        Target groupTarget;
        try {
          groupTarget = pkgValue.getPackage().getTarget(groupLabel.getName());
        } catch (NoSuchTargetException e) {
          continue;
        }
        if (groupTarget instanceof PackageGroup packageGroup) {
          if (packageGroup.contains(consumerPkg)) {
            return true;
          }
          for (Label include : packageGroup.getIncludes()) {
            if (!visitedGroups.contains(include)) {
              nextGroups.add(include);
            }
          }
        }
      }
      currentGroups = nextGroups;
    }
    return false;
  }

  /**
   * Lets {@link StarlarkOptionsParser} convert flag names to {@link Target}s through a Skyframe
   * {@link PackageValue} lookup.
   */
  private static final class SkyframeTargetLoader
      implements StarlarkOptionsParser.BuildSettingLoader {
    private final Environment env;
    private final PackageContext packageContext;

    SkyframeTargetLoader(Environment env, PackageContext packageContext) {
      this.env = env;
      this.packageContext = packageContext;
    }

    @Nullable
    @Override
    public Target loadBuildSetting(String name)
        throws InterruptedException, TargetParsingException {
      Label asLabel;
      try {
        asLabel = Label.parseWithPackageContext(name, packageContext);
      } catch (LabelSyntaxException e) {
        throw new IllegalArgumentException(e);
      }
      try {
        SkyKey pkgKey = asLabel.getPackageIdentifier();
        PackageValue pkg = (PackageValue) env.getValueOrThrow(pkgKey, NoSuchPackageException.class);
        if (pkg == null) {
          return null;
        }
        return pkg.getPackage().getTarget(asLabel.getName());
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        throw new TargetParsingException(
            String.format("Failed to load %s", name), e, DEPENDENCY_NOT_FOUND);
      }
    }
  }

  private static final class ParsedFlagsFunctionException extends SkyFunctionException {
    ParsedFlagsFunctionException(OptionsParsingException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
