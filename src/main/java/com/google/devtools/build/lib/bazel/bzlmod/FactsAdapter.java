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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.lib.json.Json;

/** Helps serialize/deserialize {@link Facts}, which contains JSON-like Starlark values. */
public class FactsAdapter extends TypeAdapter<Facts> {

  private final Gson gson = new GsonBuilder().disableHtmlEscaping().create();

  @Override
  public void write(JsonWriter out, Facts facts) throws IOException {
    String json;
    try {
      json = Json.INSTANCE.encode(facts.value());
    } catch (EvalException | InterruptedException e) {
      throw new IllegalStateException(
          "Unexpected error while serializing facts (%s): %s"
              .formatted(facts.value(), e.getMessage()),
          e);
    }
    // Round-trip the JSON through Gson to ensure it is properly indented.
    gson.toJson(gson.fromJson(json, JsonElement.class), out);
  }

  @Override
  public Facts read(JsonReader in) throws IOException {
    var jsonString = gson.toJson(JsonParser.parseReader(in));
    try (var mu = Mutability.create("FactsAdapter")) {
      var starlarkThread =
          StarlarkThread.createTransient(
              mu,
              StarlarkSemantics.builder()
                  // Ensure that UTF-8 strings are encoded correctly, matching the default semantics
                  // derived from BuildLanguageOptions.
                  .setBool(StarlarkSemantics.INTERNAL_BAZEL_ONLY_UTF_8_BYTE_STRINGS, true)
                  .build());
      return Facts.validateAndCreate(
          Json.INSTANCE.decode(jsonString, Starlark.UNBOUND, starlarkThread));
    } catch (EvalException e) {
      throw new IOException("Failed to decode facts JSON: " + e.getMessage(), e);
    }
  }
}
