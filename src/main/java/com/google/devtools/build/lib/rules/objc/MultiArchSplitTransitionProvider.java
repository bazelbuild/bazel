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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.rules.apple.AppleConfiguration.IOS_CPU_PREFIX;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PlatformRule;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * {@link TransitionFactory} implementation for multi-architecture apple rules which can accept
 * different apple platform types (such as ios or watchos).
 */
// TODO(https://github.com/bazelbuild/bazel/pull/7825): Rename to MultiArchSplitTransitionFactory.
public class MultiArchSplitTransitionProvider
    implements TransitionFactory<AttributeTransitionData>,
        SplitTransitionProviderApi,
        StarlarkValue {

  @VisibleForTesting
  static final String UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT =
      "Unsupported platform type \"%s\"";

  @VisibleForTesting
  static final String INVALID_VERSION_STRING_ERROR_FORMAT =
      "Invalid version string \"%s\". Version must be of the form 'x.y' without alphabetic "
          + "characters, such as '4.3'.";

  private static final ImmutableSet<PlatformType> SUPPORTED_PLATFORM_TYPES =
      ImmutableSet.of(
          PlatformType.IOS,
          PlatformType.WATCHOS,
          PlatformType.TVOS,
          PlatformType.MACOS,
          PlatformType.CATALYST);

  /**
   * Returns the apple platform type in the current rule context.
   *
   * @throws RuleErrorException if the platform type attribute in the current rulecontext is
   *     an invalid value
   */
  public static PlatformType getPlatformType(RuleContext ruleContext) throws RuleErrorException {
    String attributeValue =
        ruleContext.attributes().get(PlatformRule.PLATFORM_TYPE_ATTR_NAME, STRING);
    try {
      return getPlatformType(attributeValue);
    } catch (
        @SuppressWarnings("UnusedException")
        ApplePlatform.UnsupportedPlatformTypeException exception) {
      throw ruleContext.throwWithAttributeError(
          PlatformRule.PLATFORM_TYPE_ATTR_NAME,
          String.format(UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT, attributeValue));
    }
  }

  /**
   * Returns the apple platform type for the given platform type string (corresponding directly with
   * platform type attribute value).
   *
   * @throws UnsupportedPlatformTypeException if the given platform type string is not a valid type
   */
  private static PlatformType getPlatformType(String platformTypeString)
      throws ApplePlatform.UnsupportedPlatformTypeException {
    PlatformType platformType = PlatformType.fromString(platformTypeString);

    if (!SUPPORTED_PLATFORM_TYPES.contains(platformType)) {
      throw new ApplePlatform.UnsupportedPlatformTypeException(
          String.format(UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT, platformTypeString));
    } else {
      return platformType;
    }
  }

  /**
   * Validates that minimum OS was set to a valid value on the current rule.
   *
   * @throws RuleErrorException if the platform type attribute in the current rulecontext is an
   *     invalid value
   */
  public static void validateMinimumOs(RuleContext ruleContext) throws RuleErrorException {
    String attributeValue = ruleContext.attributes().get(PlatformRule.MINIMUM_OS_VERSION, STRING);
    // TODO(b/37096178): This attribute should always be a version.
    if (Strings.isNullOrEmpty(attributeValue)) {
      if (ruleContext.getFragment(AppleConfiguration.class).isMandatoryMinimumVersion()) {
        ruleContext.throwWithAttributeError(PlatformRule.MINIMUM_OS_VERSION,
            "This attribute must be explicitly specified");
      }
    } else {
      try {
        DottedVersion minimumOsVersion = DottedVersion.fromString(attributeValue);
        if (minimumOsVersion.hasAlphabeticCharacters() || minimumOsVersion.numComponents() > 2) {
          ruleContext.throwWithAttributeError(
              PlatformRule.MINIMUM_OS_VERSION,
              String.format(INVALID_VERSION_STRING_ERROR_FORMAT, attributeValue));
        }
      } catch (DottedVersion.InvalidDottedVersionException exception) {
        ruleContext.throwWithAttributeError(
            PlatformRule.MINIMUM_OS_VERSION,
            String.format(INVALID_VERSION_STRING_ERROR_FORMAT, attributeValue));
      }
    }
  }

  @Override
  public SplitTransition create(AttributeTransitionData data) {
    String platformTypeString = data.attributes().get(PlatformRule.PLATFORM_TYPE_ATTR_NAME, STRING);
    String minimumOsVersionString = data.attributes().get(PlatformRule.MINIMUM_OS_VERSION, STRING);
    PlatformType platformType;
    Optional<DottedVersion> minimumOsVersion;
    try {
      platformType = getPlatformType(platformTypeString);
      // TODO(b/37096178): This should be a mandatory attribute.
      if (Strings.isNullOrEmpty(minimumOsVersionString)) {
        minimumOsVersion = Optional.absent();
      } else {
        minimumOsVersion = Optional.of(DottedVersion.fromString(minimumOsVersionString));
      }
    } catch (ApplePlatform.UnsupportedPlatformTypeException
        | DottedVersion.InvalidDottedVersionException exception) {
      // There's no opportunity to propagate exception information up cleanly at the transition
      // provider level. This should later be registered as a rule error during the initialization
      // of the rule.
      platformType = PlatformType.IOS;
      minimumOsVersion = Optional.absent();
    }

    return new AppleBinaryTransition(platformType, minimumOsVersion);
  }

  @Override
  public boolean isSplit() {
    return true;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("apple_common.multi_arch_split");
  }

  /**
   * Transition that results in one configured target per architecture specified in the
   * platform-specific cpu flag for a particular platform type (for example, --watchos_cpus for
   * watchos platform type).
   */
  @AutoCodec
  protected static class AppleBinaryTransition implements SplitTransition {

    private final PlatformType platformType;
    // TODO(b/37096178): This should be a mandatory attribute.
    private final Optional<DottedVersion> minimumOsVersion;

    public AppleBinaryTransition(PlatformType platformType,
        Optional<DottedVersion> minimumOsVersion) {
      this.platformType = platformType;
      this.minimumOsVersion = minimumOsVersion;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(
          AppleCommandLineOptions.class,
          CoreOptions.class,
          CppOptions.class,
          ObjcCommandLineOptions.class,
          PlatformOptions.class);
    }

    @Override
    public final Map<String, BuildOptions> split(
        BuildOptionsView buildOptions, EventHandler eventHandler) {
      List<String> cpus;
      DottedVersion actualMinimumOsVersion;
      ConfigurationDistinguisher configurationDistinguisher;
      switch (platformType) {
        case IOS:
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_IOS;
          actualMinimumOsVersion =
              minimumOsVersion.isPresent()
                  ? minimumOsVersion.get()
                  : DottedVersion.maybeUnwrap(
                      buildOptions.get(AppleCommandLineOptions.class).iosMinimumOs);
          cpus = buildOptions.get(AppleCommandLineOptions.class).iosMultiCpus;
          if (cpus.isEmpty()) {
            cpus =
                ImmutableList.of(
                    AppleConfiguration.iosCpuFromCpu(buildOptions.get(CoreOptions.class).cpu));
          }
          if (actualMinimumOsVersion != null
              && actualMinimumOsVersion.compareTo(DottedVersion.fromStringUnchecked("11.0")) >= 0) {
            List<String> non32BitCpus =
                cpus.stream()
                    .filter(cpu -> !ApplePlatform.is32Bit(PlatformType.IOS, cpu))
                    .collect(Collectors.toList());
            if (!non32BitCpus.isEmpty()) {
              // TODO(b/65969900): Throw an exception here. Ideally, there would be an applicable
              // exception to throw during configuration creation, but instead this validation needs
              // to be deferred to later.
              cpus = non32BitCpus;
            }
          }
          break;
        case WATCHOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).watchosCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_WATCHOS;
          actualMinimumOsVersion = minimumOsVersion.isPresent() ? minimumOsVersion.get()
              : DottedVersion.maybeUnwrap(
                  buildOptions.get(AppleCommandLineOptions.class).watchosMinimumOs);
          break;
        case TVOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).tvosCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_TVOS_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_TVOS;
          actualMinimumOsVersion = minimumOsVersion.isPresent() ? minimumOsVersion.get()
              : DottedVersion.maybeUnwrap(
                  buildOptions.get(AppleCommandLineOptions.class).tvosMinimumOs);
          break;
        case MACOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).macosCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_MACOS_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_MACOS;
          actualMinimumOsVersion = minimumOsVersion.isPresent() ? minimumOsVersion.get()
              : DottedVersion.maybeUnwrap(
                  buildOptions.get(AppleCommandLineOptions.class).macosMinimumOs);
          break;
        case CATALYST:
          cpus = buildOptions.get(AppleCommandLineOptions.class).catalystCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_CATALYST_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_CATALYST;
          actualMinimumOsVersion =
              minimumOsVersion.isPresent()
                  ? minimumOsVersion.get()
                  : DottedVersion.maybeUnwrap(
                      buildOptions.get(AppleCommandLineOptions.class).iosMinimumOs);
          break;
        default:
          throw new IllegalArgumentException("Unsupported platform type " + platformType);
      }

      // There may be some duplicate flag values.
      cpus = ImmutableSortedSet.copyOf(cpus).asList();
      ImmutableMap.Builder<String, BuildOptions> splitBuildOptions = ImmutableMap.builder();
      for (String cpu : cpus) {
        BuildOptionsView splitOptions = buildOptions.clone();

        AppleCommandLineOptions appleCommandLineOptions =
            splitOptions.get(AppleCommandLineOptions.class);

        appleCommandLineOptions.applePlatformType = platformType;
        appleCommandLineOptions.appleSplitCpu = cpu;
        // If the new configuration does not use the apple crosstool, then it needs ios_cpu to be
        // to decide architecture.
        // TODO(b/29355778, b/28403953): Use a crosstool for any apple rule. Deprecate ios_cpu.
        appleCommandLineOptions.iosCpu = cpu;

        if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
          // Only set the (CC-compilation) CPU for dependencies if explicitly required by the user.
          // This helps users of the iOS rules who do not depend on CC rules as these CPU values
          // require additional flags to work (e.g. a custom crosstool) which now only need to be
          // set if this feature is explicitly requested.
          String platformCpu = ApplePlatform.cpuStringForTarget(platformType, cpu);
          AppleCrosstoolTransition.setAppleCrosstoolTransitionConfiguration(buildOptions,
              splitOptions, platformCpu);
        }
        switch (platformType) {
          case IOS:
          case CATALYST:
            appleCommandLineOptions.iosMinimumOs = DottedVersion.option(actualMinimumOsVersion);
            break;
          case WATCHOS:
            appleCommandLineOptions.watchosMinimumOs = DottedVersion.option(actualMinimumOsVersion);
            break;
          case TVOS:
            appleCommandLineOptions.tvosMinimumOs = DottedVersion.option(actualMinimumOsVersion);
            break;
          case MACOS:
            appleCommandLineOptions.macosMinimumOs = DottedVersion.option(actualMinimumOsVersion);
            break;
        }

        appleCommandLineOptions.configurationDistinguisher = configurationDistinguisher;
        splitBuildOptions.put(IOS_CPU_PREFIX + cpu, splitOptions.underlying());
      }
      return splitBuildOptions.build();
    }
  }
}
