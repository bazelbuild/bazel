// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.TreeMap;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.lib.json.Json;

/** Helps serialize/deserialize {@link Facts}, which contains JSON-like Starlark values. */
public class FactsAdapter extends TypeAdapter<Facts> {

  private final Gson gson = new GsonBuilder().disableHtmlEscaping().create();
  private final StarlarkSemantics semantics =
      StarlarkSemantics.builder()
          // Ensure that UTF-8 strings are encoded correctly, matching the default semantics derived
          // from BuildLanguageOptions.
          .setBool(StarlarkSemantics.INTERNAL_BAZEL_ONLY_UTF_8_BYTE_STRINGS, true)
          .build();

  @Override
  public void write(JsonWriter out, Facts facts) throws IOException {
    String json;
    try (var mu = Mutability.create("FactsAdapter")) {
      json = Json.INSTANCE.encode(facts.value(), StarlarkThread.createTransient(mu, semantics));
    } catch (EvalException | InterruptedException e) {
      throw new IllegalStateException(
          "Unexpected error while serializing facts (%s): %s"
              .formatted(facts.value(), e.getMessage()),
          e);
    }
    // Round-trip the JSON through Gson to ensure it is properly indented.
    gson.toJson(gson.fromJson(internalToUnicode(json), JsonElement.class), out);
  }

  @Override
  public Facts read(JsonReader in) throws IOException {
    if (!(JsonParser.parseReader(in) instanceof JsonObject object)) {
      throw new JsonParseException("Failed to decode facts JSON: expected an object");
    }
    try {
      return Facts.createUnchecked(readDict(object, 0));
    } catch (EvalException e) {
      throw new JsonParseException("Failed to decode facts JSON: " + e.getMessage(), e);
    }
  }

  private static Dict<String, Object> readDict(JsonObject object, int depth) throws EvalException {
    var entries = new TreeMap<String, Object>();
    for (var entry : object.entrySet()) {
      // Gson omits null object members when serializing facts, but retains null array elements.
      if (!entry.getValue().isJsonNull()) {
        entries.put(unicodeToInternal(entry.getKey()), readValue(entry.getValue(), depth + 1));
      }
    }
    return Dict.immutableCopyOf(entries);
  }

  private static Object readValue(JsonElement value, int depth) throws EvalException {
    Facts.checkDepth(depth);
    if (value instanceof JsonObject object) {
      return readDict(object, depth);
    }
    if (value instanceof JsonArray array) {
      Object[] items = new Object[array.size()];
      for (int i = 0; i < items.length; i++) {
        items[i] = readValue(array.get(i), depth + 1);
      }
      return StarlarkList.immutableOf(items);
    }
    if (value.isJsonNull()) {
      return Starlark.NONE;
    }
    var primitive = value.getAsJsonPrimitive();
    if (primitive.isString()) {
      return unicodeToInternal(primitive.getAsString());
    }
    if (primitive.isBoolean()) {
      return primitive.getAsBoolean();
    }
    String number = primitive.getAsString();
    return number.indexOf('.') >= 0 || number.indexOf('e') >= 0 || number.indexOf('E') >= 0
        ? StarlarkFloat.of(primitive.getAsDouble())
        : StarlarkInt.parse(number, 10);
  }
}
