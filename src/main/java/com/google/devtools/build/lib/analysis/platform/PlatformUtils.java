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

package com.google.devtools.build.lib.analysis.platform;

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static java.util.Comparator.comparing;

import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.Platform.Property;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.util.StringEncoding;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** Utilities for accessing platform properties. */
public final class PlatformUtils {

  private static void sortPlatformProperties(Platform.Builder builder) {
    // Internal strings sort by UTF-8 bytes, which is the code point order required by the REAPI.
    List<Platform.Property> properties =
        Ordering.from(
                comparing(Platform.Property::getName, comparing(StringEncoding::unicodeToInternal)))
            .sortedCopy(builder.getPropertiesList());
    builder.clearProperties();
    builder.addAllProperties(properties);
  }

  @Nullable
  public static Platform buildPlatformProto(Map<String, String> executionProperties) {
    if (executionProperties.isEmpty()) {
      return null;
    }
    Platform.Builder builder = Platform.newBuilder();
    for (Map.Entry<String, String> keyValue : executionProperties.entrySet()) {
      Property property =
          Property.newBuilder()
              .setName(internalToUnicode(keyValue.getKey()))
              .setValue(internalToUnicode(keyValue.getValue()))
              .build();
      builder.addProperties(property);
    }

    sortPlatformProperties(builder);
    return builder.build();
  }

  @Nullable
  public static Platform getPlatformProto(Spawn spawn, @Nullable RemoteOptions remoteOptions)
      throws UserExecException {
    return getPlatformProto(spawn, remoteOptions, ImmutableMap.of());
  }

  /**
   * Returns whether the exec property with the given key is an execution requirement that only
   * affects how Bazel itself schedules, caches or runs a spawn.
   */
  private static boolean isBazelOnlyExecutionRequirement(String key) {
    // Requirements on the execution environment, such as "requires-*", "block-*" and
    // "resources:*", are kept since remote executors may honor them.
    return key.startsWith("no-")
        || key.startsWith("supports-")
        || key.startsWith("disable-")
        || key.equals(ExecutionRequirements.LOCAL)
        || key.equals(ExecutionRequirements.WORKER_KEY_MNEMONIC);
  }

  private static Map<String, String> withoutBazelOnlyExecutionRequirements(
      Map<String, String> execProperties) {
    return Maps.filterKeys(execProperties, key -> !isBazelOnlyExecutionRequirement(key));
  }

  @Nullable
  public static Platform getPlatformProto(
      Spawn spawn, @Nullable RemoteOptions remoteOptions, Map<String, String> additionalProperties)
      throws UserExecException {
    SortedMap<String, String> defaultExecProperties =
        remoteOptions != null
            ? remoteOptions.getRemoteDefaultExecProperties()
            : ImmutableSortedMap.of();
    Map<String, String> combinedExecProperties =
        withoutBazelOnlyExecutionRequirements(spawn.getCombinedExecProperties());
    Map<String, String> platformExecProperties =
        spawn.getExecutionPlatform() != null
            ? withoutBazelOnlyExecutionRequirements(spawn.getExecutionPlatform().execProperties())
            : ImmutableMap.of();

    Map<String, String> properties = new HashMap<>();
    if (!combinedExecProperties.isEmpty()) {
      // Apply default exec properties if the execution platform does not already set
      // exec_properties
      if (platformExecProperties.isEmpty()) {
        properties.putAll(defaultExecProperties);
      }
      properties.putAll(combinedExecProperties);
    } else if (!platformExecProperties.isEmpty()) {
      properties.putAll(platformExecProperties);
    } else {
      properties.putAll(defaultExecProperties);
    }
    properties.putAll(additionalProperties);

    if (properties.isEmpty()) {
      // Execution platform is null or functionally empty
      return null;
    }

    Platform.Builder platformBuilder = Platform.newBuilder();
    for (Map.Entry<String, String> entry : properties.entrySet()) {
      platformBuilder
          .addPropertiesBuilder()
          .setName(internalToUnicode(entry.getKey()))
          .setValue(internalToUnicode(entry.getValue()));
    }
    sortPlatformProperties(platformBuilder);
    return platformBuilder.build();
  }
}
