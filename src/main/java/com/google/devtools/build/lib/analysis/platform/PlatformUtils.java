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
import com.google.common.base.Strings;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Utilities for accessing platform properties. */
public final class PlatformUtils {

  @Nullable
  public static Platform getPlatformProto(
      @Nullable PlatformInfo executionPlatform, @Nullable RemoteOptions remoteOptions)
      throws UserExecException {
    String defaultPlatformProperties = null;
    if (remoteOptions != null) {
      defaultPlatformProperties = remoteOptions.remoteDefaultPlatformProperties;
    }

    if (executionPlatform == null && Strings.isNullOrEmpty(defaultPlatformProperties)) {
      return null;
    }

    Platform.Builder platformBuilder = Platform.newBuilder();

    if (executionPlatform != null && !executionPlatform.execProperties().isEmpty()) {
      for (Map.Entry<String, String> entry : executionPlatform.execProperties().entrySet()) {
        platformBuilder.addPropertiesBuilder().setName(entry.getKey()).setValue(entry.getValue());
      }
    } else if (executionPlatform != null
        && !Strings.isNullOrEmpty(executionPlatform.remoteExecutionProperties())) {
      // Try and get the platform info from the execution properties.
      try {
        TextFormat.getParser()
            .merge(executionPlatform.remoteExecutionProperties(), platformBuilder);
      } catch (ParseException e) {
        throw new UserExecException(
            String.format(
                "Failed to parse remote_execution_properties from platform %s",
                executionPlatform.label()),
            e);
      }
    } else if (!Strings.isNullOrEmpty(defaultPlatformProperties)) {
      // Try and use the provided default value.
      try {
        TextFormat.getParser().merge(defaultPlatformProperties, platformBuilder);
      } catch (ParseException e) {
        throw new UserExecException(
            String.format(
                "Failed to parse --remote_default_platform_properties %s",
                defaultPlatformProperties),
            e);
      }
    }

    // Sort the properties.
    List<Platform.Property> properties =
        Ordering.from(Comparator.comparing(Platform.Property::getName))
            .sortedCopy(platformBuilder.getPropertiesList());
    platformBuilder.clearProperties();
    platformBuilder.addAllProperties(properties);
    return platformBuilder.build();
  }
}
