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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PlatformRule;
import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

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
  public TransitionType transitionType() {
    return TransitionType.ATTRIBUTE;
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
  protected static final class AppleBinaryTransition implements SplitTransition {

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
      AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);
      if (appleOptions.incompatibleUseToolchainResolution) {
        List<Label> platformsToSplit = appleOptions.applePlatforms;
        if (platformsToSplit.isEmpty()) {
          // If --apple_platforms is unset, instead use only the first value from --platforms.
          Label targetPlatform =
              Iterables.getFirst(buildOptions.get(PlatformOptions.class).platforms, null);
          platformsToSplit = ImmutableList.of(targetPlatform);
        }
        return MultiArchBinarySupport.handleApplePlatforms(
            buildOptions, platformType, minimumOsVersion, platformsToSplit);
      }
      return MultiArchBinarySupport.handleAppleCpus(buildOptions, platformType, minimumOsVersion);
    }
  }
}
