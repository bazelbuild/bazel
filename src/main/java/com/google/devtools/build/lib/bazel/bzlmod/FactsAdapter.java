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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;
import com.google.gson.Strictness;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import com.google.gson.stream.MalformedJsonException;
import java.io.EOFException;
import java.io.IOException;
import java.util.LinkedHashMap;
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
    gson.toJson(gson.fromJson(json, JsonElement.class), out);
  }

  @Override
  public Facts read(JsonReader in) throws IOException {
    // Match JsonParser.parseReader's behavior when called with a standalone JsonReader.
    Strictness strictness = in.getStrictness();
    if (strictness == Strictness.LEGACY_STRICT) {
      in.setStrictness(Strictness.LENIENT);
    }
    try (var mu = Mutability.create("FactsAdapter")) {
      Object value;
      boolean isEmpty = true;
      try {
        in.peek();
        isEmpty = false;
        value = readValue(in, mu);
      } catch (EOFException e) {
        // JsonParser treated an empty document as JSON null.
        if (!isEmpty) {
          throw new JsonSyntaxException(e);
        }
        value = Starlark.NONE;
      } catch (MalformedJsonException e) {
        throw new JsonSyntaxException(e);
      } catch (IOException e) {
        throw new JsonIOException(e);
      }
      return Facts.validateAndCreate(value);
    } catch (EvalException e) {
      throw new IOException("Failed to decode facts JSON: " + e.getMessage(), e);
    } catch (StackOverflowError e) {
      throw new IOException("Failed to decode facts JSON: nesting depth limit exceeded", e);
    } finally {
      in.setStrictness(strictness);
    }
  }

  private static Object readValue(JsonReader in, Mutability mu) throws IOException, EvalException {
    return switch (in.peek()) {
      case NULL -> {
        in.nextNull();
        yield Starlark.NONE;
      }
      case BOOLEAN -> in.nextBoolean();
      case STRING -> readString(in.nextString());
      case NUMBER -> {
        String number = in.nextString();
        try {
          yield number.indexOf('.') >= 0 || number.indexOf('e') >= 0 || number.indexOf('E') >= 0
              ? StarlarkFloat.of(Double.parseDouble(number))
              : StarlarkInt.parse(number, 10);
        } catch (NumberFormatException e) {
          throw Starlark.errorf("invalid number: %s", number);
        }
      }
      case BEGIN_ARRAY -> {
        in.beginArray();
        var list = StarlarkList.newList(mu);
        while (in.hasNext()) {
          list.addElement(readValue(in, mu));
        }
        in.endArray();
        yield list;
      }
      case BEGIN_OBJECT -> {
        in.beginObject();
        var entries = new LinkedHashMap<String, Object>();
        while (in.hasNext()) {
          entries.put(in.nextName(), readValue(in, mu));
        }
        in.endObject();
        // Gson kept the last value for each JSON name, then omitted null object members during
        // serialization. Decode the names after handling duplicates because readString can make
        // two distinct JSON names equal, and Json.decode kept the last of those in Gson's order.
        Dict<String, Object> dict = Dict.of(mu);
        for (var entry : entries.entrySet()) {
          if (entry.getValue() != Starlark.NONE) {
            dict.putEntry(readString(entry.getKey()), entry.getValue());
          }
        }
        yield dict;
      }
      default -> throw new IllegalStateException("Unexpected token: " + in.peek());
    };
  }

  private static String readString(String value) {
    // Gson emitted all other non-ASCII chars literally, and Json.decode left literals unchanged.
    // Gson always escaped these two characters; Json.decode converted those escapes to UTF-8
    // bytes. Preserve that representation for facts written by Json.encode as well as edited JSON.
    return value.replace("\u2028", "\u00e2\u0080\u00a8").replace("\u2029", "\u00e2\u0080\u00a9");
  }
}
