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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.TreeMap;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.Types;

/**
 * A container for user-provided JSON-like data attached to a module extension that is persisted
 * across reevaluations of the extension.
 */
@AutoValue
@AutoCodec
@StarlarkBuiltin(
    name = "Facts",
    doc =
        """
        User-provided data attached to a module extension that is persisted across reevaluations of
        the extension.

        This type supports dict-like access (e.g. `facts["key"]` and `facts.get("key")`) as well as
        membership tests (e.g. `"key" in facts`). It does not support iteration or methods like
        `keys()`, `items()`, or `len()`.
        """,
    category = DocCategory.BUILTIN)
public abstract class Facts implements StarlarkIndexable {
  public static final Facts EMPTY = new AutoValue_Facts(Dict.empty());

  public abstract Dict<String, Object> value();

  public static Facts validateAndCreate(Object value) throws EvalException {
    return new AutoValue_Facts(
        validateAndNormalize(Dict.cast(value, String.class, Object.class, "facts")));
  }

  @AutoCodec.Instantiator
  @VisibleForSerialization
  public static Facts createUnchecked(Dict<String, Object> value) {
    return new AutoValue_Facts(value);
  }

  // This limit only exists to prevent pathological uses of facts, which are meant to be
  // human-readable and friendly to VCS merges.
  private static final int MAX_FACTS_DEPTH = 7;

  @SuppressWarnings("unchecked")
  private static Dict<String, Object> validateAndNormalize(Dict<String, Object> facts)
      throws EvalException {
    return (Dict<String, Object>) validateAndNormalize(facts, MAX_FACTS_DEPTH);
  }

  private static Object validateAndNormalize(Object facts, int remainingDepth)
      throws EvalException {
    if (remainingDepth < 0) {
      throw Starlark.errorf("Facts cannot be nested more than %s levels deep", MAX_FACTS_DEPTH);
    }
    // Only permit types that can be serialized to JSON and ensure that they contain no information
    // not captured by equality by sorting dicts.
    return switch (facts) {
      case String s -> s;
      case NoneType n -> n;
      case Boolean b -> b;
      case StarlarkFloat f -> f;
      case StarlarkInt i -> i;
      case StarlarkList<?> list -> {
        Object[] normalizedList = new Object[list.size()];
        for (int i = 0; i < list.size(); i++) {
          normalizedList[i] = validateAndNormalize(list.get(i), remainingDepth - 1);
        }
        yield StarlarkList.immutableOf(normalizedList);
      }
      case Tuple tuple -> {
        // Turn a tuple into a list since JSON does not have a tuple type.
        Object[] normalizedList = new Object[tuple.size()];
        for (int i = 0; i < tuple.size(); i++) {
          normalizedList[i] = validateAndNormalize(tuple.get(i), remainingDepth - 1);
        }
        yield StarlarkList.immutableOf(normalizedList);
      }
      case Dict<?, ?> dict -> {
        var builder = new TreeMap<String, Object>();
        for (var entry : dict.entrySet()) {
          if (!(entry.getKey() instanceof String string)) {
            throw Starlark.errorf(
                "Facts keys must be strings, got '%s' (%s)",
                Starlark.repr(entry), Starlark.type(entry.getKey()));
          }
          builder.put(string, validateAndNormalize(entry.getValue(), remainingDepth - 1));
        }
        yield Dict.immutableCopyOf(builder);
      }
      default ->
          throw Starlark.errorf(
              "'%s' (%s) is not supported in facts", Starlark.repr(facts), Starlark.type(facts));
    };
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) {
    return value().get(key);
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) {
    return value().containsKey(key);
  }

  @StarlarkMethod(
      name = "get",
      doc = "Returns the value for <code>key</code> if it exists, or <code>default</code>.",
      parameters = {
        @Param(name = "key", doc = "The key to look up.", named = true),
        @Param(
            name = "default",
            doc = "The value to return if <code>key</code> is not present.",
            named = true,
            defaultValue = "None"),
      })
  public Object get(String key, Object defaultValue) throws EvalException {
    return value().getOrDefault(key, defaultValue);
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    // Don't leak the contents to Starlark.
    printer.append("Facts(<opaque, inspect with print()>)");
  }

  @Override
  public void debugPrint(Printer printer, StarlarkThread thread) {
    // Print the contents for debugging purposes.
    printer.append("Facts(");
    value().repr(printer, thread.getSemantics());
    printer.append(")");
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void checkHashable() throws EvalException {
    throw Starlark.errorf("unhashable type: '%s'", Starlark.type(this));
  }

  @Override
  public StarlarkType getStarlarkType() {
    // TODO: Use Mapping instead of dict when available.
    return Types.dict(Types.STR, Types.ANY);
  }
}
