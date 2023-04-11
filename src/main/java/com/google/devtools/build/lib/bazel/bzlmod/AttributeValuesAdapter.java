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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
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

  @Override
  public void write(JsonWriter out, AttributeValues attributeValues) throws IOException {
    JsonObject jsonObject = new JsonObject();
    for (Map.Entry<String, Object> entry : attributeValues.attributes().entrySet()) {
      jsonObject.add(entry.getKey(), serializeObject(entry.getValue()));
    }
    out.jsonValue(jsonObject.toString());
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
  private JsonElement serializeObject(Object obj) {
    if (obj.equals(Starlark.NONE)) {
      return JsonNull.INSTANCE;
    } else if (obj instanceof Boolean) {
      return new JsonPrimitive((Boolean) obj);
    } else if (obj instanceof StarlarkInt) {
      try {
        return new JsonPrimitive(((StarlarkInt) obj).toInt("serialization into the lockfile"));
      } catch (EvalException e) {
        throw new IllegalArgumentException("Unable to parse StarlarkInt to Integer: " + e);
      }
    } else if (obj instanceof String || obj instanceof Label) {
      return new JsonPrimitive(serializeObjToString(obj));
    } else if (obj instanceof Dict) {
      JsonObject jsonObject = new JsonObject();
      for (Map.Entry<?, ?> entry : ((Dict<?, ?>) obj).entrySet()) {
        jsonObject.add(serializeObjToString(entry.getKey()), serializeObject(entry.getValue()));
      }
      return jsonObject;
    } else if (obj instanceof StarlarkList) {
      JsonArray jsonArray = new JsonArray();
      for (Object item : (StarlarkList<?>) obj) {
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
      return StarlarkList.copyOf(Mutability.IMMUTABLE, list);
    } else {
      throw new IllegalArgumentException("Unsupported JSON element: " + json);
    }
  }

  /**
   * Serializes an object (Label or String) to String A label is converted to a String as it is. A
   * String is being modified with a delimiter to be easily differentiated from the label when
   * deserializing.
   *
   * @param obj String or Label
   * @return serialized object
   */
  private String serializeObjToString(Object obj) {
    if (obj instanceof Label) {
      return ((Label) obj).getUnambiguousCanonicalForm();
    }
    return "--" + obj;
  }

  /**
   * Deserializes a string to either a label or a String depending on the prefix. A string will have
   * a delimiter at the start, else it is converted to a label.
   *
   * @param value String to be deserialized
   * @return Object of type String of Label
   */
  private Object deserializeStringToObject(String value) {
    if (value.startsWith("--")) {
      return value.substring(2);
    }
    return Label.parseCanonicalUnchecked(value);
  }
}
