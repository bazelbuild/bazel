// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException3;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** Helper class that looks up {@link ToolchainTypeInfo} data. */
public class ToolchainTypeLookupUtil {

  @Nullable
  public static ImmutableMap<Label, ToolchainTypeInfo> resolveToolchainTypes(
      Environment env,
      Iterable<ConfiguredTargetKey> toolchainTypeKeys,
      boolean sanityCheckConfiguration)
      throws InterruptedException, InvalidToolchainTypeException {
    Map<
            SkyKey,
            ValueOrException3<
                ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>>
        values =
            env.getValuesOrThrow(
                toolchainTypeKeys,
                ConfiguredValueCreationException.class,
                NoSuchThingException.class,
                ActionConflictException.class);
    boolean valuesMissing = env.valuesMissing();
    Map<Label, ToolchainTypeInfo> results = valuesMissing ? null : new HashMap<>();
    for (ConfiguredTargetKey key : toolchainTypeKeys) {
      Label originalLabel = key.getLabel();
      ToolchainTypeInfo toolchainTypeInfo =
          findToolchainTypeInfo(key, values.get(key), sanityCheckConfiguration);
      if (!valuesMissing && toolchainTypeInfo != null) {
        // These are only different if the toolchain type was aliased.
        results.put(originalLabel, toolchainTypeInfo);
        results.put(toolchainTypeInfo.typeLabel(), toolchainTypeInfo);
      }
    }
    if (valuesMissing) {
      return null;
    }

    return ImmutableMap.copyOf(results);
  }

  @Nullable
  private static ToolchainTypeInfo findToolchainTypeInfo(
      ConfiguredTargetKey key,
      ValueOrException3<
              ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>
          valueOrException,
      boolean sanityCheckConfiguration)
      throws InvalidToolchainTypeException {

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
      if (sanityCheckConfiguration && !configurationKey.getFragments().isEmpty()) {
        // No fragments may be present on a toolchain_type rule in retroactive
        // trimming mode.
        String extraFragmentDescription =
            configurationKey.getFragments().stream()
                .map(cl -> cl.getSimpleName())
                .collect(joining(","));
        throw new InvalidToolchainTypeException(
            configuredTarget.getLabel(),
            "has configuration fragments, "
                + "which is forbidden in retroactive trimming mode: "
                + "extra fragments are ["
                + extraFragmentDescription
                + "]");
      }
      ToolchainTypeInfo toolchainTypeInfo = PlatformProviderUtils.toolchainType(configuredTarget);
      if (toolchainTypeInfo == null) {
        if (PlatformProviderUtils.declaredToolchainInfo(configuredTarget) != null) {
          throw new InvalidToolchainTypeException(
              configuredTarget.getLabel(),
              "is a toolchain instance. Is the rule definition for the target you're building "
                  + "setting \"toolchains =\" to a toolchain() instead of the expected "
                  + "toolchain_type()?");
        }
        throw new InvalidToolchainTypeException(configuredTarget.getLabel());
      }

      return toolchainTypeInfo;
    } catch (ConfiguredValueCreationException e) {
      throw new InvalidToolchainTypeException(key.getLabel(), e);
    } catch (NoSuchThingException e) {
      throw new InvalidToolchainTypeException(key.getLabel(), e);
    } catch (ActionConflictException e) {
      throw new InvalidToolchainTypeException(key.getLabel(), e);
    }
  }

  /** Exception used when a toolchain type label is not a valid toolchain type. */
  public static final class InvalidToolchainTypeException extends ToolchainException {
    private static final String DEFAULT_ERROR = "does not provide ToolchainTypeInfo";

    InvalidToolchainTypeException(Label label) {
      super(formatError(label, DEFAULT_ERROR));
    }

    InvalidToolchainTypeException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label, DEFAULT_ERROR), e);
    }

    public InvalidToolchainTypeException(Label label, NoSuchThingException e) {
      // Just propagate the inner exception, because it's directly actionable.
      super(e);
    }

    public InvalidToolchainTypeException(Label label, ActionConflictException e) {
      super(formatError(label, DEFAULT_ERROR), e);
    }

    InvalidToolchainTypeException(Label label, String error) {
      super(formatError(label, error));
    }

    private static String formatError(Label label, String error) {
      return String.format("Target %s was referenced as a toolchain type, but %s", label, error);
    }
  }
}
