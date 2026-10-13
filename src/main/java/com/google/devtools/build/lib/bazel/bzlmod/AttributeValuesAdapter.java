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

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.Strictness;
import com.google.gson.ToNumberPolicy;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
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
    Strictness strictness = in.getStrictness();
    if (strictness == Strictness.LEGACY_STRICT) {
      // JsonParser.parseReader treated LEGACY_STRICT readers as lenient.
      in.setStrictness(Strictness.LENIENT);
    }
    try {
      Map<String, Object> attributes = new LinkedHashMap<>();
      in.beginObject();
      while (in.hasNext()) {
        // The set of valid attribute names per repo rule is small and static, so interning them
        // helps reduce memory usage.
        attributes.put(in.nextName().intern(), deserializeObject(in));
      }
      in.endObject();
      for (Object value : attributes.values()) {
        if (value instanceof RuntimeException error) {
          throw error;
        }
      }
      return AttributeValues.create(Dict.immutableCopyOf(attributes));
    } finally {
      in.setStrictness(strictness);
    }
  }

  /**
   * Starlark Object Types Bool Integer String Label List (Int, label, string) Dict (String,list) &
   * (Label, String)
   */
  private JsonElement serializeObject(Object obj) {
    if (obj.equals(Starlark.NONE)) {
      return JsonNull.INSTANCE;
    } else if (obj instanceof Boolean bool) {
      return new JsonPrimitive(bool);
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
    } else if (obj instanceof Iterable) {
      // ListType supports any kind of Iterable, including Tuples and StarlarkLists. All of them
      // are converted to an equivalent StarlarkList during deserialization.
      JsonArray jsonArray = new JsonArray();
      for (Object item : (Iterable<?>) obj) {
        jsonArray.add(serializeObject(item));
      }
      return jsonArray;
    } else {
      throw new IllegalArgumentException("Unsupported type: " + obj.getClass());
    }
  }

  // Return conversion errors until the enclosing raw JSON object has discarded duplicate names.
  // JsonReader errors must still throw immediately, even in a value that would be discarded.
  private Object deserializeObject(JsonReader in) throws IOException {
    return switch (in.peek()) {
      case NULL -> {
        in.nextNull();
        yield Starlark.NONE;
      }
      case BOOLEAN -> in.nextBoolean();
      // Like JsonPrimitive.getAsInt(), preserve the low 32 bits and truncate fractional numbers.
      case NUMBER -> {
        Number number = ToNumberPolicy.LAZILY_PARSED_NUMBER.readNumber(in);
        try {
          yield StarlarkInt.of(number.intValue());
        } catch (NumberFormatException e) {
          yield e;
        }
      }
      case STRING -> deserializeStringToObject(in.nextString());
      case BEGIN_OBJECT -> {
        // Collapse duplicate JSON names before decoding names, as JsonObject did. Distinct JSON
        // names can decode to the same Starlark key (for example, "foo" and "'foo'").
        Map<String, Object> members = new LinkedHashMap<>();
        in.beginObject();
        while (in.hasNext()) {
          members.put(in.nextName(), deserializeObject(in));
        }
        in.endObject();
        Map<Object, Object> dict = new LinkedHashMap<>();
        for (var entry : members.entrySet()) {
          Object key = deserializeStringToObject(entry.getKey());
          if (key instanceof RuntimeException) {
            yield key;
          }
          if (entry.getValue() instanceof RuntimeException) {
            yield entry.getValue();
          }
          dict.put(key, entry.getValue());
        }
        yield Dict.immutableCopyOf(dict);
      }
      case BEGIN_ARRAY -> {
        List<Object> list = new ArrayList<>();
        in.beginArray();
        while (in.hasNext()) {
          list.add(deserializeObject(in));
        }
        in.endArray();
        for (Object value : list) {
          if (value instanceof RuntimeException) {
            yield value;
          }
        }
        yield StarlarkList.immutableCopyOf(list);
      }
      default -> throw new IllegalStateException("Unexpected token: " + in.peek());
    };
  }

  @VisibleForTesting static final String STRING_ESCAPE_SEQUENCE = "'";

  /**
   * Serializes an object (Label or String) to String. A label is converted to a String as it is. A
   * String that looks like a label is escaped so that it can be differentiated from a label when
   * deserializing, otherwise it is emitted as is.
   *
   * @param obj String or Label
   * @return serialized object
   */
  private String serializeObjToString(Object obj) {
    if (obj instanceof Label label) {
      String labelString = label.getUnambiguousCanonicalForm();
      Preconditions.checkState(labelString.startsWith("@@"));
      return internalToUnicode(labelString);
    }
    String string = (String) obj;
    // Strings that start with "@@" need to be escaped to avoid being interpreted as a label. We
    // escape by wrapping the string in the escape sequence and strip one layer of this sequence
    // during deserialization, so strings that happen to already start and end with the escape
    // sequence also have to be escaped.
    if (string.startsWith("@@")
        || (string.startsWith(STRING_ESCAPE_SEQUENCE) && string.endsWith(STRING_ESCAPE_SEQUENCE))) {
      return internalToUnicode(STRING_ESCAPE_SEQUENCE + string + STRING_ESCAPE_SEQUENCE);
    }
    return internalToUnicode(string);
  }

  /**
   * Deserializes a string to either a label or a String depending on the prefix and presence of the
   * escape sequence.
   *
   * @param unicodeValue String to be deserialized
   * @return Object of type String of Label
   */
  private Object deserializeStringToObject(String unicodeValue) {
    try {
      String value = unicodeToInternal(unicodeValue);
      // A string represents a label if and only if it starts with "@@".
      if (value.startsWith("@@")) {
        return Label.parseCanonicalUnchecked(value);
      }
      // Strings that start and end with the escape sequence always require one layer to be stripped.
      if (value.startsWith(STRING_ESCAPE_SEQUENCE) && value.endsWith(STRING_ESCAPE_SEQUENCE)) {
        return value.substring(
            STRING_ESCAPE_SEQUENCE.length(), value.length() - STRING_ESCAPE_SEQUENCE.length());
      }
      return value;
    } catch (IllegalArgumentException | IndexOutOfBoundsException e) {
      return e;
    }
  }
}
