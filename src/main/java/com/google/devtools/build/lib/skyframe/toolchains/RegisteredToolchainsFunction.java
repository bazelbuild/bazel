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

package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableTable;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AliasProvider.TargetMode;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleContext.PrerequisiteValidationContext;
import com.google.devtools.build.lib.analysis.RuleContext.PrerequisiteValidator;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.platform.ToolchainRule;
import com.google.devtools.build.lib.server.FailureDetails.Toolchain.Code;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.TargetPatternUtil;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} that returns all registered toolchains available for toolchain resolution in
 * a given target configuration.
 */
public class RegisteredToolchainsFunction implements SkyFunction {

  private final PrerequisiteValidator prerequisiteValidator;

  public RegisteredToolchainsFunction(PrerequisiteValidator prerequisiteValidator) {
    this.prerequisiteValidator = prerequisiteValidator;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RegisteredToolchainsValue.Key key = (RegisteredToolchainsValue.Key) skyKey;
    BuildConfigurationValue configuration =
        (BuildConfigurationValue) env.getValue(key.getConfigurationKey());
    if (configuration == null) {
      return null;
    }

    // Expand the registered toolchains. This is shared by all configurations with the same
    // --extra_toolchains.
    ToolchainDeclarationsValue declarations;
    try {
      declarations =
          (ToolchainDeclarationsValue)
              env.getValueOrThrow(
                  ToolchainDeclarationsValue.Key.create(
                      configuration.getFragment(PlatformConfiguration.class).getExtraToolchains()),
                  InvalidToolchainLabelException.class);
    } catch (InvalidToolchainLabelException e) {
      throw new RegisteredToolchainsFunctionException(e, Transience.PERSISTENT);
    }
    if (declarations == null) {
      return null;
    }

    // Load the configured target for each, and get the declared toolchain providers.
    ImmutableList<DeclaredToolchainInfo> registeredToolchains =
        configureRegisteredToolchains(env, configuration, declarations.labels());
    if (env.valuesMissing()) {
      return null;
    }

    // Analyze the target settings in this configuration. These are typically shared by many
    // toolchains.
    Map<Label, ConfiguredTargetAndData> targetSettings =
        analyzeTargetSettings(env, key.getConfigurationKey(), registeredToolchains);
    // The target settings are validated as if they were dependencies of the toolchain, which
    // requires the toolchain's rule.
    SkyframeLookupResult toolchainPackages =
        env.getValuesAndExceptions(
            registeredToolchains.stream()
                .filter(toolchain -> !toolchain.targetSettings().isEmpty())
                .map(toolchain -> toolchain.targetLabel().getPackageIdentifier())
                .collect(toImmutableSet()));
    if (targetSettings == null || env.valuesMissing()) {
      return null;
    }

    // Check which toolchains are valid according to their configuration.
    ImmutableList.Builder<DeclaredToolchainInfo> validToolchains = new ImmutableList.Builder<>();
    // Some toolchains end up with repeated reasons, so use a HashBasedTable to handle duplicates.
    Table<Label, Label, String> rejectedToolchains = key.debug() ? HashBasedTable.create() : null;
    for (DeclaredToolchainInfo toolchain : registeredToolchains) {
      try {
        Consumer<String> errorHandler =
            key.debug()
                ? message ->
                    rejectedToolchains.put(
                        toolchain.toolchainType().typeLabel(), toolchain.targetLabel(), message)
                : null;
        if (ConfigMatchingUtil.validate(
            toolchain.targetLabel(),
            validateTargetSettings(
                toolchain, configuration, targetSettings, toolchainPackages, env.getListener()),
            errorHandler,
            ToolchainRule.TARGET_SETTING_ATTR)) {
          validToolchains.add(toolchain);
        }
      } catch (InvalidConfigurationException e) {
        throw new RegisteredToolchainsFunctionException(
            new InvalidToolchainLabelException(toolchain.targetLabel(), e), Transience.PERSISTENT);
      }
    }

    return RegisteredToolchainsValue.create(
        validToolchains.build(),
        rejectedToolchains != null ? ImmutableTable.copyOf(rejectedToolchains) : null);
  }

  /**
   * Analyzes the target settings of the given toolchains in the given configuration and returns
   * them by label, or {@code null} if Skyframe values are missing.
   */
  @Nullable
  private static Map<Label, ConfiguredTargetAndData> analyzeTargetSettings(
      Environment env,
      BuildConfigurationKey configurationKey,
      ImmutableList<DeclaredToolchainInfo> toolchains)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    SkyframeLookupResult values =
        env.getValuesAndExceptions(
            toolchains.stream()
                .flatMap(toolchain -> toolchain.targetSettings().stream())
                .map(label -> configuredTargetKey(label, configurationKey))
                .collect(toImmutableSet()));
    Map<Label, ConfiguredTargetValue> targetSettingValues = new HashMap<>();
    // Report an error for any toolchain before returning due to missing values so that it is
    // attributed to the toolchain even if the evaluation of other target settings is aborted.
    for (DeclaredToolchainInfo toolchain : toolchains) {
      for (Label label : toolchain.targetSettings()) {
        if (targetSettingValues.containsKey(label)) {
          continue;
        }
        try {
          var value =
              (ConfiguredTargetValue)
                  values.getOrThrow(
                      configuredTargetKey(label, configurationKey),
                      ConfiguredValueCreationException.class,
                      NoSuchThingException.class);
          if (value != null) {
            targetSettingValues.put(label, value);
          }
        } catch (ConfiguredValueCreationException e) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(toolchain.targetLabel(), e), Transience.PERSISTENT);
        } catch (NoSuchThingException e) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(toolchain.targetLabel(), e), Transience.PERSISTENT);
        }
      }
    }
    if (env.valuesMissing()) {
      return null;
    }

    Map<Label, ConfiguredTargetAndData> targetSettings = new HashMap<>();
    for (Map.Entry<Label, ConfiguredTargetValue> entry : targetSettingValues.entrySet()) {
      var targetSetting =
          ConfiguredTargetAndData.fromConfiguredTargetInSkyframe(entry.getValue(), env);
      if (targetSetting == null) {
        return null;
      }
      targetSettings.put(entry.getKey(), targetSetting);
    }
    return targetSettings;
  }

  private static ConfiguredTargetKey configuredTargetKey(
      Label label, BuildConfigurationKey configurationKey) {
    return ConfiguredTargetKey.builder()
        .setLabel(label)
        .setConfigurationKey(configurationKey)
        .build();
  }

  /**
   * Returns the {@link ConfigMatchingProvider}s of the target settings of the given toolchain after
   * validating them as if they were dependencies of the toolchain.
   */
  private ImmutableList<ConfigMatchingProvider> validateTargetSettings(
      DeclaredToolchainInfo toolchain,
      BuildConfigurationValue configuration,
      Map<Label, ConfiguredTargetAndData> targetSettings,
      SkyframeLookupResult toolchainPackages,
      ExtendedEventHandler eventHandler)
      throws RegisteredToolchainsFunctionException {
    if (toolchain.targetSettings().isEmpty()) {
      return ImmutableList.of();
    }
    Rule rule = getRule(toolchain.targetLabel(), toolchainPackages);
    Attribute attribute =
        rule.getRuleClassObject()
            .getAttributeProvider()
            .getAttributeByName(ToolchainRule.TARGET_SETTING_ATTR);
    var validationContext = new TargetSettingsValidationContext(rule, configuration, eventHandler);
    ImmutableList.Builder<ConfigMatchingProvider> matchingProviders = ImmutableList.builder();
    for (Label label : toolchain.targetSettings()) {
      ConfiguredTargetAndData targetSetting = targetSettings.get(label);
      ConfigMatchingProvider matchingProvider =
          targetSetting.getConfiguredTarget().getProvider(ConfigMatchingProvider.class);
      if (matchingProvider != null) {
        matchingProviders.add(matchingProvider);
      } else {
        validationContext.attributeError(
            ToolchainRule.TARGET_SETTING_ATTR,
            String.format(
                "%s is misplaced here (expected config_setting)",
                AliasProvider.describeTargetWithAliases(targetSetting, TargetMode.WITH_KIND)));
      }
      prerequisiteValidator.validate(validationContext, targetSetting, attribute);
    }
    if (validationContext.hasErrors()) {
      throw new RegisteredToolchainsFunctionException(
          new InvalidToolchainLabelException(
              toolchain.targetLabel(), "invalid " + ToolchainRule.TARGET_SETTING_ATTR),
          Transience.PERSISTENT);
    }
    return matchingProviders.build();
  }

  private static Rule getRule(Label label, SkyframeLookupResult packages) {
    PackageValue packageValue = (PackageValue) packages.get(label.getPackageIdentifier());
    try {
      return (Rule) packageValue.getPackage().getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      // The toolchain has already been analyzed successfully.
      throw new IllegalStateException(e);
    }
  }

  /**
   * Reports the errors and warnings about the target settings of a toolchain in the same way as if
   * they were dependencies of it.
   */
  private static final class TargetSettingsValidationContext
      implements PrerequisiteValidationContext {
    private final Rule rule;
    private final BuildConfigurationValue configuration;
    private final ExtendedEventHandler eventHandler;
    private boolean hasErrors = false;

    TargetSettingsValidationContext(
        Rule rule, BuildConfigurationValue configuration, ExtendedEventHandler eventHandler) {
      this.rule = rule;
      this.configuration = configuration;
      this.eventHandler = eventHandler;
    }

    @Override
    public Rule getRule() {
      return rule;
    }

    @Override
    public BuildConfigurationValue getConfiguration() {
      return configuration;
    }

    @Override
    @Nullable
    public Aspect getMainAspect() {
      return null;
    }

    @Override
    public void ruleWarning(String message) {
      eventHandler.handle(Event.warn(rule.getLocation(), prefixRuleMessage(message)));
    }

    @Override
    public void ruleError(String message) {
      hasErrors = true;
      eventHandler.handle(Event.error(rule.getLocation(), prefixRuleMessage(message)));
    }

    @Override
    public void attributeWarning(String attrName, String message) {
      eventHandler.handle(
          Event.warn(rule.getLocation(), prefixAttributeMessage(attrName, message)));
    }

    @Override
    public void attributeError(String attrName, String message) {
      hasErrors = true;
      eventHandler.handle(
          Event.error(rule.getLocation(), prefixAttributeMessage(attrName, message)));
    }

    @Override
    public boolean hasErrors() {
      return hasErrors;
    }

    private String prefixRuleMessage(String message) {
      return String.format("in %s rule %s: %s", rule.getRuleClass(), rule.getLabel(), message);
    }

    private String prefixAttributeMessage(String attrName, String message) {
      return String.format(
          "in %s attribute of %s rule %s: %s",
          attrName, rule.getRuleClass(), rule.getLabel(), message);
    }
  }

  @Nullable
  private static ImmutableList<DeclaredToolchainInfo> configureRegisteredToolchains(
      Environment env, BuildConfigurationValue configuration, Collection<Label> labels)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    ImmutableSet<ActionLookupKey> keys =
        labels.stream()
            .map(
                label ->
                    ConfiguredTargetKey.builder()
                        .setLabel(label)
                        .setConfiguration(configuration)
                        .build())
            .collect(toImmutableSet());

    SkyframeLookupResult values = env.getValuesAndExceptions(keys);
    ImmutableList.Builder<DeclaredToolchainInfo> toolchains = new ImmutableList.Builder<>();
    boolean valuesMissing = false;
    for (ActionLookupKey key : keys) {
      Label toolchainLabel = key.getLabel();
      try {
        SkyValue value = values.getOrThrow(key, ConfiguredValueCreationException.class);
        if (value == null) {
          valuesMissing = true;
          continue;
        }

        ConfiguredTarget target = ((ConfiguredTargetValue) value).getConfiguredTarget();
        DeclaredToolchainInfo toolchainInfo = PlatformProviderUtils.declaredToolchainInfo(target);
        if (toolchainInfo == null) {
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(toolchainLabel), Transience.PERSISTENT);
        }
        toolchains.add(toolchainInfo);
      } catch (ConfiguredValueCreationException e) {
        throw new RegisteredToolchainsFunctionException(
            new InvalidToolchainLabelException(toolchainLabel, e), Transience.PERSISTENT);
      }
    }

    if (valuesMissing) {
      return null;
    }
    return toolchains.build();
  }

  /**
   * Used to indicate that the given {@link Label} represents a {@link ConfiguredTarget} which is
   * not a valid {@link DeclaredToolchainInfo} provider.
   */
  public static final class InvalidToolchainLabelException extends ToolchainException {

    public InvalidToolchainLabelException(Label invalidLabel) {
      super(
          formatMessage(
              invalidLabel.getCanonicalForm(),
              "target does not provide the DeclaredToolchainInfo provider"));
    }

    public InvalidToolchainLabelException(TargetPatternUtil.InvalidTargetPatternException e) {
      this(e.getInvalidPattern(), e.getTpe());
    }

    public InvalidToolchainLabelException(String invalidPattern, TargetParsingException e) {
      super(formatMessage(invalidPattern, e.getMessage()), e);
    }

    public InvalidToolchainLabelException(Label invalidLabel, ConfiguredValueCreationException e) {
      super(formatMessage(invalidLabel.getCanonicalForm(), e.getMessage()), e);
    }

    public InvalidToolchainLabelException(Label invalidLabel, NoSuchThingException e) {
      super(formatMessage(invalidLabel.getCanonicalForm(), e.getMessage()), e);
    }

    private InvalidToolchainLabelException(Label invalidLabel, String reason) {
      super(formatMessage(invalidLabel.getCanonicalForm(), reason));
    }

    public InvalidToolchainLabelException(Label invalidLabel, InvalidConfigurationException e) {
      super(formatMessage(invalidLabel.getCanonicalForm(), e.getMessage()), e);
    }

    @Override
    protected Code getDetailedCode() {
      return Code.INVALID_TOOLCHAIN;
    }

    private static String formatMessage(String invalidPattern, String reason) {
      return StringUtil.formatNested(
          String.format("invalid registered toolchain '%s'", invalidPattern), reason);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  public static class RegisteredToolchainsFunctionException extends SkyFunctionException {

    public RegisteredToolchainsFunctionException(
        InvalidToolchainLabelException cause, Transience transience) {
      super(cause, transience);
    }

    public RegisteredToolchainsFunctionException(
        ExternalDepsException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
