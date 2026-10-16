// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.asset.v1.FetchGrpc;
import build.bazel.remote.execution.v2.ActionCacheGrpc;
import build.bazel.remote.execution.v2.CapabilitiesGrpc;
import build.bazel.remote.execution.v2.ContentAddressableStorageGrpc;
import com.google.bytestream.ByteStreamGrpc;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Type;
import java.time.Duration;
import java.util.Map;

/** Builds Bazel's generated gRPC service config for remote services. */
public final class RemoteGrpcServiceConfig {
  private static final Gson GSON = new Gson();
  private static final Type MAP_TYPE = new TypeToken<Map<String, ?>>() {}.getType();
  private static final ImmutableSet<String> SUPPORTED_TOP_LEVEL_FIELDS =
      ImmutableSet.of("methodConfig");
  private static final ImmutableSet<String> SUPPORTED_METHOD_CONFIG_FIELDS =
      ImmutableSet.of("name", "timeout");
  private static final ImmutableSet<String> SUPPORTED_NAME_FIELDS =
      ImmutableSet.of("service", "method");

  private RemoteGrpcServiceConfig() {}

  public static ImmutableMap<String, ?> create(RemoteOptions options) {
    return create(options.getRemoteTimeout());
  }

  public static ImmutableMap<String, ?> create(RemoteOptions options, Path workingDirectory)
      throws IOException {
    PathFragment serviceConfigPath = options.getRemoteGrpcServiceConfig();
    if (serviceConfigPath == null) {
      return create(options);
    }
    return parse(workingDirectory.getRelative(serviceConfigPath));
  }

  static ImmutableMap<String, ?> create(Duration remoteTimeout) {
    return ImmutableMap.of(
        "methodConfig",
        ImmutableList.of(
            ImmutableMap.of(
                "name",
                ImmutableList.of(
                    ImmutableMap.of("service", ActionCacheGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", CapabilitiesGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", ContentAddressableStorageGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", ByteStreamGrpc.SERVICE_NAME),
                    ImmutableMap.of("service", FetchGrpc.SERVICE_NAME)),
                "timeout",
                remoteTimeout.toSeconds() + "s")));
  }

  private static ImmutableMap<String, ?> parse(Path serviceConfigPath) throws IOException {
    try (Reader reader = new InputStreamReader(serviceConfigPath.getInputStream(), UTF_8)) {
      JsonObject root = requireObject(JsonParser.parseReader(reader), "service config");
      rejectUnsupportedFields(root, SUPPORTED_TOP_LEVEL_FIELDS, "service config");
      rejectUnsupportedMethodConfigFields(root);
      Map<String, ?> serviceConfig = GSON.fromJson(root, MAP_TYPE);
      return ImmutableMap.copyOf(serviceConfig);
    } catch (JsonParseException e) {
      throw new IOException(
          "failed to parse " + serviceConfigPath.getPathString() + ": " + e.getMessage(), e);
    }
  }

  private static void rejectUnsupportedMethodConfigFields(JsonObject rootObject)
      throws IOException {
    JsonElement methodConfigsElement = rootObject.get("methodConfig");
    if (methodConfigsElement == null || !methodConfigsElement.isJsonArray()) {
      return;
    }

    JsonArray methodConfigs = methodConfigsElement.getAsJsonArray();
    for (int i = 0; i < methodConfigs.size(); ++i) {
      JsonElement methodConfigElement = methodConfigs.get(i);
      if (!methodConfigElement.isJsonObject()) {
        continue;
      }

      JsonObject methodConfig = methodConfigElement.getAsJsonObject();
      String methodConfigPath = "methodConfig[" + i + "]";
      rejectUnsupportedFields(methodConfig, SUPPORTED_METHOD_CONFIG_FIELDS, methodConfigPath);
      rejectUnsupportedNameFields(methodConfig, methodConfigPath);
    }
  }

  private static void rejectUnsupportedNameFields(JsonObject methodConfig, String methodConfigPath)
      throws IOException {
    JsonElement namesElement = methodConfig.get("name");
    if (namesElement == null || !namesElement.isJsonArray()) {
      return;
    }

    JsonArray names = namesElement.getAsJsonArray();
    for (int i = 0; i < names.size(); ++i) {
      JsonElement nameElement = names.get(i);
      if (nameElement.isJsonObject()) {
        rejectUnsupportedFields(
            nameElement.getAsJsonObject(),
            SUPPORTED_NAME_FIELDS,
            methodConfigPath + ".name[" + i + "]");
      }
    }
  }

  private static void rejectUnsupportedFields(
      JsonObject object, ImmutableSet<String> supportedFields, String path) throws IOException {
    for (Map.Entry<String, JsonElement> entry : object.entrySet()) {
      if (!supportedFields.contains(entry.getKey())) {
        throw new IOException(path + " contains unsupported field '" + entry.getKey() + "'");
      }
    }
  }

  private static JsonObject requireObject(JsonElement element, String path) throws IOException {
    if (element == null || !element.isJsonObject()) {
      throw new IOException(path + " must be an object");
    }
    return element.getAsJsonObject();
  }
}
