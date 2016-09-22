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

import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.SplitTransitionProvider;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import java.util.List;

/**
 * {@link SplitTransitionProvider} implementation for multi-architecture apple rules which can
 * accept different apple platform types (such as ios or watchos).
 */
public class MultiArchSplitTransitionProvider implements SplitTransitionProvider {
  
  @VisibleForTesting
  static final String UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT =
      "Unsupported platform type \"%s\"";
  
  private static final ImmutableSet<PlatformType> SUPPORTED_PLATFORM_TYPES =
      ImmutableSet.of(PlatformType.IOS, PlatformType.WATCHOS, PlatformType.TVOS);

  /**
   * Returns the apple platform type in the current rule context.
   * 
   * @throws RuleErrorException if the platform type attribute in the current rulecontext is
   *     an invalid value
   */
  public static PlatformType getPlatformType(RuleContext ruleContext) throws RuleErrorException {
    String attributeValue =
        ruleContext.attributes().get(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME, STRING);
    try {
      return getPlatformType(attributeValue);
    } catch (IllegalArgumentException exception) {
      throw ruleContext.throwWithAttributeError(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME,
          String.format(UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT, attributeValue));
    }
  }

  /**
   * Returns the apple platform type for the given platform type string (corresponding directly
   * with platform type attribute value).
   * 
   * @throws IllegalArgumentException if the given platform type string is not a valid type
   */
  public static PlatformType getPlatformType(String platformTypeString) {
    PlatformType platformType = PlatformType.fromString(platformTypeString);

    if (!SUPPORTED_PLATFORM_TYPES.contains(platformType)) {
      throw new IllegalArgumentException(
          String.format(UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT, platformTypeString));
    } else {
      return platformType;
    }
  }

  private static final ImmutableMap<PlatformType, AppleBinaryTransition>
      SPLIT_TRANSITIONS_BY_TYPE = ImmutableMap.<PlatformType, AppleBinaryTransition>builder()
          .put(PlatformType.IOS, new AppleBinaryTransition(PlatformType.IOS))
          .put(PlatformType.WATCHOS, new AppleBinaryTransition(PlatformType.WATCHOS))
          .put(PlatformType.TVOS, new AppleBinaryTransition(PlatformType.TVOS))
          .build();

  @Override
  public SplitTransition<?> apply(Rule fromRule) {
    String platformTypeString = NonconfigurableAttributeMapper.of(fromRule)
        .get(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME, STRING);
    PlatformType platformType;
    try {
      platformType = getPlatformType(platformTypeString);
    } catch (IllegalArgumentException exception) {
      // There's no opportunity to propagate exception information up cleanly at the transition
      // provider level. This should later be registered as a rule error during the initialization
      // of the rule.
      platformType = PlatformType.IOS;
    }

    return SPLIT_TRANSITIONS_BY_TYPE.get(platformType);
  }

  /**
   * Returns the full list of potential split transitions this split transition provider may
   * produce.
   */
  public static List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
    return ImmutableList.<SplitTransition<BuildOptions>>copyOf(
        SPLIT_TRANSITIONS_BY_TYPE.values());
  }

  /**
   * Transition that results in one configured target per architecture specified in the
   * platform-specific cpu flag for a particular platform type (for example, --watchos_cpus
   * for watchos platform type).
   */
  protected static class AppleBinaryTransition implements SplitTransition<BuildOptions> {

    private final PlatformType platformType;

    public AppleBinaryTransition(PlatformType platformType) {
      this.platformType = platformType;
    }

    @Override
    public final List<BuildOptions> split(BuildOptions buildOptions) {
      List<String> cpus;
      ConfigurationDistinguisher configurationDistinguisher;
      switch (platformType) {
        case IOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).iosMultiCpus;
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_IOS;
          break;
        case WATCHOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).watchosCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_WATCHOS;
          break;
        case TVOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).tvosCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_TVOS_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_TVOS;
          break;
        default:
          throw new IllegalArgumentException("Unsupported platform type " + platformType);
      }

      ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();
      for (String cpu : cpus) {
        BuildOptions splitOptions = buildOptions.clone();

        splitOptions.get(AppleCommandLineOptions.class).applePlatformType = platformType;
        splitOptions.get(AppleCommandLineOptions.class).appleSplitCpu = cpu;
        // Set for backwards compatibility with rules that depend on this flag, even when
        // ios is not the platform type.
        // TODO(b/28958783): Clean this up.
        splitOptions.get(AppleCommandLineOptions.class).iosCpu = cpu;
        if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
          // Only set the (CC-compilation) CPU for dependencies if explicitly required by the user.
          // This helps users of the iOS rules who do not depend on CC rules as these CPU values
          // require additional flags to work (e.g. a custom crosstool) which now only need to be
          // set if this feature is explicitly requested.
          splitOptions.get(BuildConfiguration.Options.class).cpu =
              String.format("%s_%s", platformType, cpu);
        }
        splitOptions.get(AppleCommandLineOptions.class).configurationDistinguisher =
            configurationDistinguisher;
        splitBuildOptions.add(splitOptions);
      }
      return splitBuildOptions.build();
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }
  }
}