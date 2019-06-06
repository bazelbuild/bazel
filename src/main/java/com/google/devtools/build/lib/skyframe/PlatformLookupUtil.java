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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Predicates.equalTo;
import static com.google.common.base.Predicates.not;
import static java.util.stream.Collectors.joining;

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException3;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** Helper class that looks up {@link PlatformInfo} data. */
public class PlatformLookupUtil {

  @Nullable
  public static Map<ConfiguredTargetKey, PlatformInfo> getPlatformInfo(
      Iterable<ConfiguredTargetKey> platformKeys, Environment env, boolean sanityCheckConfiguration)
      throws InterruptedException, InvalidPlatformException {

    Map<
            SkyKey,
            ValueOrException3<
                ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>>
        values =
            env.getValuesOrThrow(
                platformKeys,
                ConfiguredValueCreationException.class,
                NoSuchThingException.class,
                ActionConflictException.class);
    boolean valuesMissing = env.valuesMissing();
    Map<ConfiguredTargetKey, PlatformInfo> platforms = valuesMissing ? null : new HashMap<>();
    for (ConfiguredTargetKey key : platformKeys) {
      PlatformInfo platformInfo = findPlatformInfo(key, values.get(key), sanityCheckConfiguration);
      if (!valuesMissing && platformInfo != null) {
        platforms.put(key, platformInfo);
      }
    }
    if (valuesMissing) {
      return null;
    }

    return platforms;
  }

  /**
   * Returns the {@link PlatformInfo} provider from the {@link ConfiguredTarget} in the {@link
   * ValueOrException3}, or {@code null} if the {@link ConfiguredTarget} is not present. If the
   * {@link ConfiguredTarget} does not have a {@link PlatformInfo} provider, a {@link
   * InvalidPlatformException} is thrown.
   */
  @Nullable
  private static PlatformInfo findPlatformInfo(
      ConfiguredTargetKey key,
      ValueOrException3<
              ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>
          valueOrException,
      boolean sanityCheckConfiguration)
      throws InvalidPlatformException {

    try {
      ConfiguredTargetValue ctv = (ConfiguredTargetValue) valueOrException.get();
      if (ctv == null) {
        return null;
      }

      ConfiguredTarget configuredTarget = ctv.getConfiguredTarget();
      BuildConfigurationValue.Key configurationKey = configuredTarget.getConfigurationKey();
      // This check is necessary because trimming for other rules assumes that platform resolution
      // uses the platform fragment and _only_ the platform fragment. Without this check, it's
      // possible another fragment could slip in without us realizing, and thus break this
      // assumption.
      if (sanityCheckConfiguration
          && configurationKey.getFragments().stream()
              .anyMatch(not(equalTo(PlatformConfiguration.class)))) {
        // Only the PlatformConfiguration fragment may be present on a platform rule in retroactive
        // trimming mode.
        String extraFragmentDescription =
            configurationKey.getFragments().stream()
                .filter(not(equalTo(PlatformConfiguration.class)))
                .map(cl -> cl.getSimpleName())
                .collect(joining(","));
        throw new InvalidPlatformException(
            configuredTarget.getLabel(),
            "has fragments other than PlatformConfiguration, "
                + "which is forbidden in retroactive trimming mode: "
                + "extra fragments are ["
                + extraFragmentDescription
                + "]");
      }
      PlatformInfo platformInfo = PlatformProviderUtils.platform(configuredTarget);
      if (platformInfo == null) {
        throw new InvalidPlatformException(configuredTarget.getLabel());
      }

      return platformInfo;
    } catch (ConfiguredValueCreationException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    } catch (NoSuchThingException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    } catch (ActionConflictException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    }
  }

  /** Exception used when a platform label is not a valid platform. */
  public static final class InvalidPlatformException extends ToolchainException {
    private static final String DEFAULT_ERROR = "does not provide PlatformInfo";

    InvalidPlatformException(Label label) {
      super(formatError(label, DEFAULT_ERROR));
    }

    InvalidPlatformException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label, DEFAULT_ERROR), e);
    }

    public InvalidPlatformException(Label label, NoSuchThingException e) {
      // Just propagate the inner exception, because it's directly actionable.
      super(e);
    }

    public InvalidPlatformException(Label label, ActionConflictException e) {
      super(formatError(label, DEFAULT_ERROR), e);
    }

    InvalidPlatformException(Label label, String error) {
      super(formatError(label, error));
    }

    private static String formatError(Label label, String error) {
      return String.format("Target %s was referenced as a platform, but %s", label, error);
    }
  }
}
