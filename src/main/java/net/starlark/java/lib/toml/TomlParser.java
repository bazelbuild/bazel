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

package net.starlark.java.lib.toml;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.dataformat.toml.TomlMapper;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.packages.NativeInfo;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.temporal.TemporalAccessor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkIterable;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkSet;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Structure;

// Tests at //src/test/java/net/starlark/java/eval:testdata/toml.star

/**
 * TomlParser defines the Starlark {@code toml} module, which provides functions for encoding/decoding
 */
@StarlarkBuiltin(
    name = "toml",
    category = "core.lib",
    doc = "Module toml is a Starlark module of TOML-related functions.")
public final class TomlParser implements StarlarkValue {

  private TomlParser() {}

  /**
   * The module instance. You may wish to add this to your predeclared environment under the name
   * "toml".
   */
  public static final TomlParser INSTANCE = new TomlParser();

  /**
   * Encodes a Starlark value as TOML.
   */
  @StarlarkMethod(
      name = "encode",
      doc =
          "<p>The encode function accepts one required positional argument, which it converts to"
              + " TOML by cases:\n"
              + "<ul>\n"
              + "<li>True, and False are converted to 'true', and 'false',"
              + " respectively.\n"
              + "<li>An int, no matter how large, is encoded as a decimal integer. Some decoders"
              + " may not be able to decode very large integers.\n"
              + "<li>A float is encoded using a decimal point or an exponent or both, even if its"
              + " numeric value is an integer. It is an error to encode a non-finite "
              + " floating-point value.\n"
              + "<li>A string value is encoded as a TOML string literal that denotes the value.\n"
              + "<li>A dict is encoded as a TOML table. It is an error if any key is not a string.\n"
              + "<li>A list or tuple is encoded as a TOML array.\n"
              + "</ul>\n"
              + "Encoding any other value yields an error.\n",
      parameters = {@Param(name = "x")},
      useStarlarkThread = true)
  public String encode(Object x, StarlarkThread thread) throws EvalException, InterruptedException {
    try {
      Object javaValue = convertToJava(x, thread.getSemantics());

      // TOML requires the root to be a map/table structure
      if (!(javaValue instanceof Map)) {
        throw Starlark.errorf("TOML encode requires a dict at the top level, got %s", Starlark.type(x));
      }

      // Empty map produces empty TOML document
      if (((Map<?, ?>) javaValue).isEmpty()) {
        return "";
      }

      TomlMapper mapper = new TomlMapper();
      return mapper.writeValueAsString(javaValue);
    } catch (JsonProcessingException e) {
      throw Starlark.errorf("TOML encode error: %s", e.getMessage());
    } catch (StackOverflowError unused) {
      throw Starlark.errorf("nesting depth limit exceeded");
    }
  }

  /** Parses a TOML string as a Starlark value. */
  @StarlarkMethod(
      name = "decode",
      doc =
          "The decode function has one required positional parameter: a TOML string.\n"
              + "It returns the Starlark value that the string denotes.\n"
              + "<code>\"true\"</code> and <code>\"false\"</code>"
              + " are parsed as <code>True</code>, and <code>False</code>.\n"
              + "<li>Numbers are parsed as int, or as a float if they contain a decimal point or an"
              + " exponent. TOML date values are decoded as strings. "
              + "If <code>x</code> is not a valid TOML encoding and the optional"
              + " <code>default</code> parameter is specified (including specified as"
              + " <code>None</code>), this function returns the <code>default</code> value.\n"
              + "If <code>x</code> is not a valid TOML encoding and the optional"
              + " <code>default</code> parameter is <em>not</em> specified, this function fails.",
      parameters = {
        @Param(name = "x", doc = "TOML string to decode."),
        @Param(
            name = "default",
            named = true,
            doc = "If specified, the value to return when <code>x</code> cannot be decoded.",
            defaultValue = "unbound")
      },
      useStarlarkThread = true)
  public Object decode(String x, Object defaultValue, StarlarkThread thread) throws EvalException {
    TomlMapper mapper = new TomlMapper();
    try {
      Object result = mapper.readValue(x, Object.class);
      return convertToStarlark(result, null);
    } catch (JsonProcessingException e) {
      if (defaultValue != Starlark.UNBOUND) {
        return defaultValue;
      }
      throw Starlark.errorf("TOML decode error: %s", e.getMessage());
    }
  }

  private static Object convertToStarlark(Object x, @Nullable Mutability mutability) throws EvalException {
    if (x == null) {
      return Starlark.NONE;
    } else if (Starlark.valid(x)) {
      return x;
    } else if (x instanceof TemporalAccessor) {
      // Jackson returns various java.time types for TOML dates
      return x.toString();
    } else if (x instanceof Number) {
      if (x instanceof Integer) {
        return StarlarkInt.of((Integer) x);
      } else if (x instanceof Long) {
        return StarlarkInt.of((Long) x);
      } else if (x instanceof BigInteger) {
        return StarlarkInt.of((BigInteger) x);
      } else if (x instanceof Double) {
        return StarlarkFloat.of((double) x);
      } else if (x instanceof Float) {
        return StarlarkFloat.of(((Float) x).doubleValue());
      } else if (x instanceof BigDecimal) {
        return StarlarkFloat.of(((BigDecimal) x).doubleValue());
      }
    } else if (x instanceof List) {
      List<?> list = (List<?>) x;
      List<Object> converted = new ArrayList<>(list.size());
      for (Object elem : list) {
        converted.add(convertToStarlark(elem, mutability));
      }
      return StarlarkList.copyOf(mutability, converted);
    } else if (x instanceof Map) {
      Map<?, ?> map = (Map<?, ?>) x;
      Map<Object, Object> converted = new LinkedHashMap<>(map.size());
      for (Map.Entry<?, ?> entry : map.entrySet()) {
        converted.put(
            convertToStarlark(entry.getKey(), mutability),
            convertToStarlark(entry.getValue(), mutability));
      }
      return Dict.copyOf(mutability, converted);
    } else if (x instanceof Set) {
      Set<?> set = (Set<?>) x;
      Set<Object> converted = new LinkedHashSet<>(set.size());
      for (Object elem : set) {
        converted.add(convertToStarlark(elem, mutability));
      }
      return StarlarkSet.copyOf(mutability, converted);
    }
    throw Starlark.errorf("invalid Starlark value: %s",
                          x.getClass() == null ? "null" : x.getClass());
  }

  /**
   * Converts a Starlark value to a Java object suitable for TOML encoding.
   */
  private static Object convertToJava(Object x, StarlarkSemantics semantics) throws EvalException, InterruptedException {
    if (x == Starlark.NONE) {
      throw Starlark.errorf("cannot encode None as TOML");
    }

    if (x instanceof Boolean || x instanceof String) {
      return x;
    }

    if (x instanceof StarlarkInt) {
      return ((StarlarkInt) x).toBigInteger();
    }

    if (x instanceof StarlarkFloat) {
      double val = ((StarlarkFloat) x).toDouble();
      if (!Double.isFinite(val)) {
        throw Starlark.errorf("cannot encode non-finite float %s", x);
      }
      return val;
    }

    // e.g. dict (must have string keys)
    if (x instanceof Map<?, ?> m) {
      // Sort keys for determinism
      Object[] keys = m.keySet().toArray();
      for (Object key : keys) {
        if (!(key instanceof String)) {
          throw Starlark.errorf(
              "%s has %s key, want string", Starlark.type(x), Starlark.type(key));
        }
      }
      Arrays.sort(keys);

      Map<String, Object> result = new LinkedHashMap<>();
      for (Object key : keys) {
        try {
          result.put((String) key, convertToJava(m.get(key), semantics));
        } catch (EvalException ex) {
          throw Starlark.errorf(
              "in %s key %s: %s", Starlark.type(x), Starlark.repr(key), ex.getMessage());
        }
      }
      return result;
    }

    // e.g. set (sort for determinism)
    if (x instanceof StarlarkSet) {
      List<Object> elements = new ArrayList<>();
      for (Object elem : (StarlarkSet<?>) x) {
        elements.add(elem);
      }
      // Sort elements for deterministic output
      Object[] sorted = elements.toArray();
      Arrays.sort(sorted);

      List<Object> result = new ArrayList<>();
      int i = 0;
      for (Object elem : sorted) {
        try {
          result.add(convertToJava(elem, semantics));
        } catch (EvalException ex) {
          throw Starlark.errorf("at %s index %d: %s", Starlark.type(x), i, ex.getMessage());
        }
        i++;
      }
      return result;
    }

    // e.g. tuple, list
    if (x instanceof StarlarkIterable<?> it) {
      List<Object> result = new ArrayList<>();
      int i = 0;
      for (Object elem : it) {
        try {
          result.add(convertToJava(elem, semantics));
        } catch (EvalException ex) {
          throw Starlark.errorf("at %s index %d: %s", Starlark.type(x), i, ex.getMessage());
        }
        i++;
      }
      return result;
    }

    // e.g. struct
    if (x instanceof Structure || x instanceof NativeInfo) {
      // Sort fields for determinism
      List<String> fields =
          Ordering.natural().sortedCopy(Starlark.dir(Mutability.IMMUTABLE, semantics, x));

      Map<String, Object> result = new LinkedHashMap<>();
      for (String field : fields) {
        try {
          Object v =
              Starlark.getattr(
                  Mutability.IMMUTABLE,
                  semantics,
                  x,
                  field,
                  null); // may fail (field not defined)
          // Skip callables (methods)
          if (x instanceof NativeInfo && v instanceof StarlarkCallable) {
            continue;
          }
          result.put(field, convertToJava(v, semantics));
        } catch (EvalException ex) {
          throw Starlark.errorf("in %s field .%s: %s", Starlark.type(x), field, ex.getMessage());
        }
      }
      return result;
    }

    throw Starlark.errorf("cannot encode %s as TOML", Starlark.type(x));
  }
}
