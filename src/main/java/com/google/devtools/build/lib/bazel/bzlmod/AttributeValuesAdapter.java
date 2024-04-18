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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;

/** Helps serialize/deserialize {@link AttributeValues}, which contains Starlark values. */
public class AttributeValuesAdapter extends TypeAdapter<AttributeValues> {

  private final Gson gson = new GsonBuilder().disableHtmlEscaping().create();

  @Override
  public void write(JsonWriter out, AttributeValues attributeValues) throws IOException {
    out.beginObject();
    for (Map.Entry<String, Object> entry : attributeValues.attributes().entrySet()) {
      out.name(entry.getKey());
      gson.toJson(serializeObject(entry.getValue()), out);
    }
    out.endObject();
  }

  @Override
  public AttributeValues read(JsonReader in) throws IOException {
    JsonObject jsonObject = JsonParser.parseReader(in).getAsJsonObject();
    Dict.Builder<String, Object> dict = Dict.builder();
    for (Map.Entry<String, JsonElement> entry : jsonObject.entrySet()) {
      dict.put(entry.getKey(), deserializeObject(entry.getValue()));
    }
    return AttributeValues.create(dict.buildImmutable());
  }

  /**
   * Starlark Object Types Bool Integer String Label List (Int, label, string) Dict (String,list) &
   * (Label, String)
   */
  private JsonElement serializeObject(Object obj) throws IOException {
    if (obj.equals(Starlark.NONE)) {
      return JsonNull.INSTANCE;
    } else if (obj instanceof Boolean bool) {
      return new JsonPrimitive(bool);
    } else if (obj instanceof StarlarkInt i) {
      try {
        return new JsonPrimitive(i.toInt("serialization into the lockfile"));
      } catch (EvalException e) {
        throw new IllegalArgumentException("Unable to parse StarlarkInt to Integer: " + e);
      }
    } else if (obj instanceof String s) {
      return new JsonPrimitive(serializeString(s));
    } else if (obj instanceof Label label) {
      return new JsonPrimitive(serializeLabel(label));
    } else if (obj instanceof Dict<?, ?> dict) {
      JsonObject jsonObject = new JsonObject();
      for (Map.Entry<?, ?> entry : dict.entrySet()) {
        String key =
            entry.getKey() instanceof Label label
                ? serializeLabel(label)
                : serializeString((String) entry.getKey());
        jsonObject.add(key, serializeObject(entry.getValue()));
      }
      return jsonObject;
    } else if (obj instanceof Iterable<?> iter) {
      // ListType supports any kind of Iterable, including Tuples and StarlarkLists. All of them
      // are converted to an equivalent StarlarkList during deserialization.
      JsonArray jsonArray = new JsonArray();
      for (Object item : iter) {
        jsonArray.add(serializeObject(item));
      }
      return jsonArray;
    } else {
      throw new IllegalArgumentException("Unsupported type: " + obj.getClass());
    }
  }

  private Object deserializeObject(JsonElement json) {
    if (json == null || json.isJsonNull()) {
      return Starlark.NONE;
    } else if (json.isJsonPrimitive()) {
      JsonPrimitive jsonPrimitive = json.getAsJsonPrimitive();
      if (jsonPrimitive.isBoolean()) {
        return jsonPrimitive.getAsBoolean();
      } else if (jsonPrimitive.isNumber()) {
        return StarlarkInt.of(jsonPrimitive.getAsInt());
      } else if (jsonPrimitive.isString()) {
        return deserializeStringToObject(jsonPrimitive.getAsString());
      } else {
        throw new IllegalArgumentException("Unsupported JSON primitive: " + jsonPrimitive);
      }
    } else if (json.isJsonObject()) {
      JsonObject jsonObject = json.getAsJsonObject();
      Dict.Builder<Object, Object> dict = Dict.builder();
      for (Map.Entry<String, JsonElement> entry : jsonObject.entrySet()) {
        dict.put(deserializeStringToObject(entry.getKey()), deserializeObject(entry.getValue()));
      }
      return dict.buildImmutable();
    } else if (json.isJsonArray()) {
      JsonArray jsonArray = json.getAsJsonArray();
      List<Object> list = new ArrayList<>();
      for (JsonElement item : jsonArray) {
        list.add(deserializeObject(item));
      }
      return StarlarkList.immutableCopyOf(list);
    } else {
      throw new IllegalArgumentException("Unsupported JSON element: " + json);
    }
  }

  @VisibleForTesting static final String STRING_ESCAPE_SEQUENCE = "'";

  private String serializeLabel(Label label) throws IOException {
    if (!label.getRepository().isVisible()) {
      throw new IOException("invalid label found in repo attributes: " + label);
    }
    String s = label.getUnambiguousCanonicalForm();
    Preconditions.checkState(s.startsWith("@@"));
    return s;
  }

  /**
   * Serializes a string. A string that looks like a label is escaped so that it can be
   * differentiated from a label when deserializing (since labels are also serialized as strings),
   * otherwise it is emitted as is.
   */
  private String serializeString(String s) {
    // Strings that start with "@@" need to be escaped to avoid being interpreted as a label. We
    // escape by wrapping the string in the escape sequence and strip one layer of this sequence
    // during deserialization, so strings that happen to already start and end with the escape
    // sequence also have to be escaped.
    if (s.startsWith("@@")
        || (s.startsWith(STRING_ESCAPE_SEQUENCE) && s.endsWith(STRING_ESCAPE_SEQUENCE))) {
      return STRING_ESCAPE_SEQUENCE + s + STRING_ESCAPE_SEQUENCE;
    }
    return s;
  }

  /**
   * Deserializes a string to either a label or a String depending on the prefix and presence of the
   * escape sequence.
   *
   * @param value String to be deserialized
   * @return Object of type String of Label
   */
  private Object deserializeStringToObject(String value) {
    // A string represents a label if and only if it starts with "@@".
    if (value.startsWith("@@")) {
      try {
        return Label.parseCanonical(value);
      } catch (LabelSyntaxException e) {
        throw new JsonParseException(
            String.format("error parsing label \"%s\" from the lockfile", value), e);
      }
    }
    // Strings that start and end with the escape sequence always require one layer to be stripped.
    if (value.startsWith(STRING_ESCAPE_SEQUENCE) && value.endsWith(STRING_ESCAPE_SEQUENCE)) {
      return value.substring(
          STRING_ESCAPE_SEQUENCE.length(), value.length() - STRING_ESCAPE_SEQUENCE.length());
    }
    return value;
  }
}
