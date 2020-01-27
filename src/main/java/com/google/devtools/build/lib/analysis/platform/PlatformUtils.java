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

import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.Platform.Property;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import javax.annotation.Nullable;

/** Utilities for accessing platform properties. */
public final class PlatformUtils {

  private static void sortPlatformProperties(Platform.Builder builder) {
    List<Platform.Property> properties =
        Ordering.from(Comparator.comparing(Platform.Property::getName))
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
          Property.newBuilder().setName(keyValue.getKey()).setValue(keyValue.getValue()).build();
      builder.addProperties(property);
    }

    sortPlatformProperties(builder);
    return builder.build();
  }

  @Nullable
  public static Platform getPlatformProto(Spawn spawn, @Nullable RemoteOptions remoteOptions)
      throws UserExecException {
    SortedMap<String, String> defaultExecProperties =
        remoteOptions != null
            ? remoteOptions.getRemoteDefaultExecProperties()
            : ImmutableSortedMap.of();

    if (spawn.getExecutionPlatform() == null
        && spawn.getCombinedExecProperties().isEmpty()
        && defaultExecProperties.isEmpty()) {
      return null;
    }

    Platform.Builder platformBuilder = Platform.newBuilder();

    if (!spawn.getCombinedExecProperties().isEmpty()) {
      for (Map.Entry<String, String> entry : spawn.getCombinedExecProperties().entrySet()) {
        platformBuilder.addPropertiesBuilder().setName(entry.getKey()).setValue(entry.getValue());
      }
    } else if (spawn.getExecutionPlatform() != null
        && !Strings.isNullOrEmpty(spawn.getExecutionPlatform().remoteExecutionProperties())) {
      // Try and get the platform info from the execution properties.
      try {
        TextFormat.getParser()
            .merge(spawn.getExecutionPlatform().remoteExecutionProperties(), platformBuilder);
      } catch (ParseException e) {
        throw new UserExecException(
            String.format(
                "Failed to parse remote_execution_properties from platform %s",
                spawn.getExecutionPlatform().label()),
            e);
      }
    } else {
      for (Map.Entry<String, String> property : defaultExecProperties.entrySet()) {
        platformBuilder.addProperties(
            Property.newBuilder().setName(property.getKey()).setValue(property.getValue()).build());
      }
    }

    sortPlatformProperties(platformBuilder);
    return platformBuilder.build();
  }
}
