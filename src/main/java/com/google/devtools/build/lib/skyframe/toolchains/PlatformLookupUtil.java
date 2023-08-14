// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.server.FailureDetails.Toolchain.Code;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** Helper class that looks up {@link PlatformInfo} data. */
public class PlatformLookupUtil {

  @Nullable
  public static Map<ConfiguredTargetKey, PlatformInfo> getPlatformInfo(
      ImmutableList<ConfiguredTargetKey> platformKeys, Environment env)
      throws InterruptedException, InvalidPlatformException {
    validatePlatformKeys(platformKeys, env);
    if (env.valuesMissing()) {
      return null;
    }

    SkyframeLookupResult values = env.getValuesAndExceptions(platformKeys);
    boolean valuesMissing = env.valuesMissing();
    Map<ConfiguredTargetKey, PlatformInfo> platforms = valuesMissing ? null : new HashMap<>();
    for (ConfiguredTargetKey key : platformKeys) {
      PlatformInfo platformInfo = findPlatformInfo(key, values);
      if (!valuesMissing && platformInfo != null) {
        platforms.put(key, platformInfo);
      }
    }
    if (valuesMissing) {
      return null;
    }

    return platforms;
  }

  /** Validate that all keys are for actual platform targets. */
  private static void validatePlatformKeys(
      ImmutableList<ConfiguredTargetKey> platformKeys, Environment env)
      throws InterruptedException, InvalidPlatformException {
    // Load the packages. This should already be in Skyframe and thus not require a restart.
    ImmutableSet<PackageIdentifier> packageKeys =
        platformKeys.stream()
            .map(ConfiguredTargetKey::getLabel)
            .map(Label::getPackageIdentifier)
            .collect(toImmutableSet());

    SkyframeLookupResult values = env.getValuesAndExceptions(packageKeys);
    boolean valuesMissing = env.valuesMissing();
    Map<PackageIdentifier, Package> packages = valuesMissing ? null : new HashMap<>();
    for (PackageIdentifier packageKey : packageKeys) {
      try {
        PackageValue packageValue =
            (PackageValue) values.getOrThrow(packageKey, NoSuchPackageException.class);
        if (!valuesMissing && packageValue != null) {
          packages.put(packageKey, packageValue.getPackage());
        }
      } catch (NoSuchPackageException e) {
        throw new InvalidPlatformException(e);
      }
    }
    if (env.valuesMissing()) {
      if (valuesMissing != env.valuesMissing()) {
        BugReport.sendBugReport(
            new IllegalStateException(
                "Some value from " + packageKeys + " was missing, this should never happen"));
      }
      return;
    }

    // Now check each platform.
    for (ConfiguredTargetKey platformKey : platformKeys) {
      try {
        Label platformLabel = platformKey.getLabel();
        Target target =
            packages.get(platformLabel.getPackageIdentifier()).getTarget(platformLabel.getName());
        if (!hasPlatformInfo(target)) {
          // validation failure
          throw new InvalidPlatformException(platformLabel);
        }
      } catch (NoSuchTargetException e) {
        throw new InvalidPlatformException(e);
      }
    }
  }

  /**
   * Returns the {@link PlatformInfo} provider from the {@link ConfiguredTarget} in the {@link
   * SkyframeLookupResult}, or {@code null} if the {@link ConfiguredTarget} is not present. If the
   * {@link ConfiguredTarget} does not have a {@link PlatformInfo} provider, a {@link
   * InvalidPlatformException} is thrown.
   */
  @Nullable
  private static PlatformInfo findPlatformInfo(ConfiguredTargetKey key, SkyframeLookupResult values)
      throws InvalidPlatformException {
    try {
      ConfiguredTargetValue ctv =
          (ConfiguredTargetValue)
              values.getOrThrow(
                  key,
                  ConfiguredValueCreationException.class,
                  NoSuchThingException.class,
                  ActionConflictException.class);
      if (ctv == null) {
        return null;
      }

      ConfiguredTarget configuredTarget = ctv.getConfiguredTarget();
      PlatformInfo platformInfo = PlatformProviderUtils.platform(configuredTarget);
      if (platformInfo == null) {
        throw new InvalidPlatformException(configuredTarget.getLabel());
      }

      return platformInfo;
    } catch (ConfiguredValueCreationException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    } catch (NoSuchThingException e) {
      throw new InvalidPlatformException(e);
    } catch (ActionConflictException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    }
  }

  public static boolean hasPlatformInfo(Target target) {
    Rule rule = target.getAssociatedRule();
    // If the rule uses toolchain resolution, it can't be used as a target or exec platform.
    if (rule == null) {
      return false;
    }
    if (rule.useToolchainResolution()) {
      return false;
    }

    return rule.getRuleClassObject()
        .getAdvertisedProviders()
        .advertises(PlatformInfo.PROVIDER.id());
  }

  /** Exception used when a platform label is not a valid platform. */
  public static final class InvalidPlatformException extends ToolchainException {
    private static final String DEFAULT_ERROR = "does not provide PlatformInfo";

    public InvalidPlatformException(Label label) {
      super(formatError(label, DEFAULT_ERROR));
    }

    public InvalidPlatformException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label, DEFAULT_ERROR), e);
    }

    public InvalidPlatformException(NoSuchThingException e) {
      // Just propagate the inner exception, because it's directly actionable.
      super(e);
    }

    public InvalidPlatformException(Label label, ActionConflictException e) {
      super(formatError(label, DEFAULT_ERROR), e);
    }

    @Override
    protected Code getDetailedCode() {
      return Code.INVALID_PLATFORM_VALUE;
    }

    private static String formatError(Label label, String error) {
      return String.format("Target %s was referenced as a platform, but %s", label, error);
    }
  }
}
