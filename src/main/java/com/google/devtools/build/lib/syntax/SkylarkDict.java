// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkMutable.MutableMap;

import java.util.Map;
import java.util.TreeMap;

import javax.annotation.Nullable;

/**
 * Skylark Dict module.
 */
@SkylarkModule(name = "dict", doc =
    "A language built-in type to support dicts. "
    + "Example of dict literal:<br>"
    + "<pre class=\"language-python\">d = {\"a\": 2, \"b\": 5}</pre>"
    + "Use brackets to access elements:<br>"
    + "<pre class=\"language-python\">e = d[\"a\"]   # e == 2</pre>"
    + "Dicts support the <code>+</code> operator to concatenate two dicts. In case of multiple "
    + "keys the second one overrides the first one. Examples:<br>"
    + "<pre class=\"language-python\">"
    + "d = {\"a\" : 1} + {\"b\" : 2}   # d == {\"a\" : 1, \"b\" : 2}\n"
    + "d += {\"c\" : 3}              # d == {\"a\" : 1, \"b\" : 2, \"c\" : 3}\n"
    + "d = d + {\"c\" : 5}           # d == {\"a\" : 1, \"b\" : 2, \"c\" : 5}</pre>"
    + "Iterating on a dict is equivalent to iterating on its keys (in sorted order).<br>"
    + "Dicts support the <code>in</code> operator, testing membership in the keyset of the dict. "
    + "Example:<br>"
    + "<pre class=\"language-python\">\"a\" in {\"a\" : 2, \"b\" : 5}   # evaluates as True"
    + "</pre>")
public final class SkylarkDict<K, V>
    extends MutableMap<K, V> implements Map<K, V> {

  private TreeMap<K, V> contents = new TreeMap<>(EvalUtils.SKYLARK_COMPARATOR);

  private Mutability mutability;

  @Override
  public Mutability mutability() {
    return mutability;
  }

  private SkylarkDict(@Nullable Environment env) {
    mutability = env == null ? Mutability.IMMUTABLE : env.mutability();
  }

  /** @return a dict mutable in given environment only */
  public static <K, V> SkylarkDict<K, V> of(@Nullable Environment env) {
    return new SkylarkDict<>(env);
  }

  /** @return a dict mutable in given environment only, with given initial key and value */
  public static <K, V> SkylarkDict<K, V> of(@Nullable Environment env, K k, V v) {
    return SkylarkDict.<K, V>of(env).putUnsafe(k, v);
  }

  /** @return a dict mutable in given environment only, with two given initial key value pairs */
  public static <K, V> SkylarkDict<K, V> of(
      @Nullable Environment env, K k1, V v1, K k2, V v2) {
    return SkylarkDict.<K, V>of(env).putUnsafe(k1, v1).putUnsafe(k2, v2);
  }

  /** @return a dict mutable in given environment only, with contents copied from given map */
  public static <K, V> SkylarkDict<K, V> copyOf(
      @Nullable Environment env, Map<? extends K, ? extends V> m) {
    return SkylarkDict.<K, V>of(env).putAllUnsafe(m);
  }

  private SkylarkDict<K, V> putUnsafe(K k, V v) {
    contents.put(k, v);
    return this;
  }

  private <KK extends K, VV extends V> SkylarkDict putAllUnsafe(Map<KK, VV> m) {
    for (Map.Entry<KK, VV> e : m.entrySet()) {
      contents.put(e.getKey(), e.getValue());
    }
    return this;
  }

  /**
   * The underlying contents is a (usually) mutable data structure.
   * Read access is forwarded to these contents.
   * This object must not be modified outside an {@link Environment}
   * with a correct matching {@link Mutability},
   * which should be checked beforehand using {@link #checkMutable}.
   * it need not be an instance of {@link com.google.common.collect.ImmutableMap}.
   */
  @Override
  protected Map<K, V> getContentsUnsafe() {
    return contents;
  }

  public void put(K k, V v, Location loc, Environment env) throws EvalException {
    checkMutable(loc, env);
    EvalUtils.checkValidDictKey(k);
    contents.put(k, v);
  }

  public void putAll(Map<? extends K, ? extends V> m, Location loc, Environment env)
      throws EvalException {
    checkMutable(loc, env);
    putAllUnsafe(m);
  }

  // Other methods
  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.printList(buffer, entrySet(), "{", ", ", "}", null, quotationMark);
  }

  /**
   * Cast a SkylarkDict to a {@code Map<K, V>} after checking its current contents.
   * Treat None as meaning the empty ImmutableMap.
   * @param obj the Object to cast. null and None are treated as an empty list.
   * @param keyType the expected class of keys
   * @param valueType the expected class of values
   * @param description a description of the argument being converted, or null, for debugging
   */
  public static <K, V> SkylarkDict<K, V> castSkylarkDictOrNoneToDict(
      Object obj, Class<K> keyType, Class<V> valueType, @Nullable String description)
      throws EvalException {
    if (EvalUtils.isNullOrNone(obj)) {
      return empty();
    }
    if (obj instanceof SkylarkDict) {
      return ((SkylarkDict<?, ?>) obj).getContents(keyType, valueType, description);
    }
    throw new EvalException(null,
        Printer.format("Illegal argument: %s is not of expected type dict or NoneType",
            description == null ? Printer.repr(obj) : String.format("'%s'", description)));
  }

  /**
   * Cast a SkylarkDict to a {@code SkylarkDict<K, V>} after checking its current contents.
   * @param keyType the expected class of keys
   * @param valueType the expected class of values
   * @param description a description of the argument being converted, or null, for debugging
   */
  @SuppressWarnings("unchecked")
  public <KeyType, ValueType> SkylarkDict<KeyType, ValueType> getContents(
      Class<KeyType> keyType, Class<ValueType> valueType, @Nullable String description)
      throws EvalException {
    Object keyDescription = description == null
        ? null : Printer.formattable("'%s' key", description);
    Object valueDescription = description == null
        ? null : Printer.formattable("'%s' value", description);
    for (Map.Entry<?, ?> e : this.entrySet()) {
      SkylarkType.checkType(e.getKey(), keyType, keyDescription);
      SkylarkType.checkType(e.getValue(), valueType, valueDescription);
    }
    return (SkylarkDict<KeyType, ValueType>) this;
  }


  public static <K, V> SkylarkDict<K, V> plus(
      SkylarkDict<? extends K, ? extends V> left,
      SkylarkDict<? extends K, ? extends V> right,
      @Nullable Environment env) {
    SkylarkDict<K, V> result = SkylarkDict.<K, V>of(env);
    result.putAllUnsafe(left);
    result.putAllUnsafe(right);
    return result;
  }

  private static final SkylarkDict<?, ?> EMPTY = of(null);

  public static <K, V> SkylarkDict<K, V> empty() {
    return (SkylarkDict<K, V>) EMPTY;
  }
}
