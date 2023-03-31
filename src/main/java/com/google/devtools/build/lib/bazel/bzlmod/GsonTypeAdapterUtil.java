// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.devtools.build.lib.bazel.bzlmod.DelegateTypeAdapterFactory.DICT;
import static com.google.devtools.build.lib.bazel.bzlmod.DelegateTypeAdapterFactory.IMMUTABLE_BIMAP;
import static com.google.devtools.build.lib.bazel.bzlmod.DelegateTypeAdapterFactory.IMMUTABLE_LIST;
import static com.google.devtools.build.lib.bazel.bzlmod.DelegateTypeAdapterFactory.IMMUTABLE_MAP;

import com.google.common.base.Splitter;
import com.google.common.base.VerifyException;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

/**
 * Utility class to hold type adapters and helper methods to get gson registered with type adapters
 */
public final class GsonTypeAdapterUtil {

  public static final TypeAdapter<Version> VERSION_TYPE_ADAPTER =
      new TypeAdapter<>() {
        @Override
        public void write(JsonWriter jsonWriter, Version version) throws IOException {
          jsonWriter.value(version.toString());
        }

        @Override
        public Version read(JsonReader jsonReader) throws IOException {
          Version version;
          String versionString = jsonReader.nextString();
          try {
            version = Version.parse(versionString);
          } catch (ParseException e) {
            throw new JsonParseException(
                String.format("Unable to parse Version %s from the lockfile", versionString), e);
          }
          return version;
        }
      };

  public static final TypeAdapter<ModuleKey> MODULE_KEY_TYPE_ADAPTER =
      new TypeAdapter<>() {
        @Override
        public void write(JsonWriter jsonWriter, ModuleKey moduleKey) throws IOException {
          jsonWriter.value(moduleKey.toString());
        }

        @Override
        public ModuleKey read(JsonReader jsonReader) throws IOException {
          String jsonString = jsonReader.nextString();
          if (jsonString.equals("<root>")) {
            return ModuleKey.ROOT;
          }
          List<String> parts = Splitter.on('@').splitToList(jsonString);
          if (parts.get(1).equals("_")) {
            return ModuleKey.create(parts.get(0), Version.EMPTY);
          }

          Version version;
          try {
            version = Version.parse(parts.get(1));
          } catch (ParseException e) {
            throw new JsonParseException(
                String.format("Unable to parse ModuleKey %s version from the lockfile", jsonString),
                e);
          }
          return ModuleKey.create(parts.get(0), version);
        }
      };

  public static TypeAdapter<Registry> registryTypeAdapter(RegistryFactory registryFactory) {
    return new TypeAdapter<>() {
      @Override
      public void write(JsonWriter jsonWriter, Registry registry) throws IOException {
        jsonWriter.value(registry.getUrl());
      }

      @Override
      public Registry read(JsonReader jsonReader) throws IOException {
        try {
          return registryFactory.getRegistryWithUrl(jsonReader.nextString());
        } catch (URISyntaxException e) {
          throw new VerifyException("Lockfile registry URL is not valid", e);
        }
      }
    };
  }

  private static final GsonBuilder adapterGson =
      new GsonBuilder()
          .setPrettyPrinting()
          .registerTypeAdapterFactory(GenerateTypeAdapter.FACTORY)
          .registerTypeAdapterFactory(DICT)
          .registerTypeAdapterFactory(IMMUTABLE_MAP)
          .registerTypeAdapterFactory(IMMUTABLE_LIST)
          .registerTypeAdapterFactory(IMMUTABLE_BIMAP)
          .registerTypeAdapter(Version.class, VERSION_TYPE_ADAPTER)
          .registerTypeAdapter(ModuleKey.class, MODULE_KEY_TYPE_ADAPTER);

  /**
   * Gets a gson with registered adapters needed to read the lockfile.
   *
   * @param registryFactory Registry factory to use in the registry adapter
   * @return gson with type adapters
   */
  public static Gson getLockfileGsonWithTypeAdapters(RegistryFactory registryFactory) {
    return adapterGson
        .registerTypeAdapter(Registry.class, registryTypeAdapter(registryFactory))
        .create();
  }

  private GsonTypeAdapterUtil() {}
}
