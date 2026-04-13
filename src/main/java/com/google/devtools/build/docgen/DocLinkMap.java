// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.docgen;

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.ImmutableMap;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Represents a link mapping that acts as input to {@link RuleLinkExpander} and {@link
 * SourceUrlMapper}.
 */
public record DocLinkMap(
    // For RuleLinkExpander
    String beRoot,
    ImmutableMap<String, String> beReferences,
    String sourceUrlRoot, // For SourceUrlMapper
    ImmutableMap<String, String> repoPathRewrites) {

  public static DocLinkMap createFromFile(String filePath) {
    try {
      return GSON.fromJson(Files.readString(Path.of(filePath)), DocLinkMap.class);
    } catch (IOException | JsonSyntaxException ex) {
      throw new IllegalArgumentException("Failed to read link map from " + filePath, ex);
    }
  }

  private static final JsonDeserializer<ImmutableMap<String, String>> IMMUTABLE_MAP_DESERIALIZER =
      (jsonElement, unusedType, unusedContext) ->
          jsonElement.getAsJsonObject().entrySet().stream()
              .collect(
                  toImmutableMap(entry -> entry.getKey(), entry -> entry.getValue().getAsString()));

  private static final Gson GSON =
      new GsonBuilder()
          .registerTypeAdapter(
              new TypeToken<ImmutableMap<String, String>>() {}.getType(),
              IMMUTABLE_MAP_DESERIALIZER)
          .create();
}
