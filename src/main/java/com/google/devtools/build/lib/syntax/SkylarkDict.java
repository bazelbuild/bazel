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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.SkylarkMutable.MutableMap;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A Skylark dictionary (dict).
 *
 * <p>Although this implements the {@link Map} interface, it is not mutable via that interface's
 * methods. Instead, use the mutators that take in a {@link Mutability} object.
 */
@SkylarkModule(
  name = "dict",
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "A language built-in type representating a dictionary (associative mapping). "
          + "Dictionaries may be constructed with a special literal syntax:<br>"
          + "<pre class=\"language-python\">d = {\"a\": 2, \"b\": 5}</pre>"
          + "See also the <a href=\"globals.html#dict\">dict()</a> constructor function. "
          + "When using the literal syntax, it is an error to have duplicated keys. "
          + "Use square brackets to access elements:<br>"
          + "<pre class=\"language-python\">e = d[\"a\"]   # e == 2</pre>"
          + "Like lists, they can also be constructed using a comprehension syntax:<br>"
          + "<pre class=\"language-python\">d = {i: 2*i for i in range(20)}\n"
          + "e = d[8]       # e == 16</pre>"
          + "Dictionaries are mutable. You can add new elements or mutate existing ones:"
          + "<pre class=\"language-python\">d[\"key\"] = 5</pre>"
          + "<p>Iterating over a dict is equivalent to iterating over its keys. The "
          + "<code>in</code> operator tests for membership in the keyset of the dict.<br>"
          + "<pre class=\"language-python\">\"a\" in {\"a\" : 2, \"b\" : 5} "
          + "# evaluates as True</pre>"
          + "The iteration order for a dict is deterministic but not specified."
)
public final class SkylarkDict<K, V> extends MutableMap<K, V>
    implements Map<K, V>, SkylarkIndexable {

  private final LinkedHashMap<K, V> contents = new LinkedHashMap<>();

  private final Mutability mutability;

  private SkylarkDict(@Nullable Mutability mutability) {
    this.mutability = mutability == null ? Mutability.IMMUTABLE : mutability;
  }

  private SkylarkDict(@Nullable Environment env) {
    this.mutability = env == null ? Mutability.IMMUTABLE : env.mutability();
  }

  private static final SkylarkDict<?, ?> EMPTY = withMutability(Mutability.IMMUTABLE);

  /** Returns an immutable empty dict. */
  // Safe because the empty singleton is immutable.
  @SuppressWarnings("unchecked")
  public static <K, V> SkylarkDict<K, V> empty() {
    return (SkylarkDict<K, V>) EMPTY;
  }

  /** Returns an empty dict with the given {@link Mutability}. */
  public static <K, V> SkylarkDict<K, V> withMutability(@Nullable Mutability mutability) {
    return new SkylarkDict<>(mutability);
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

  // TODO(bazel-team): Make other methods that take in mutabilities instead of environments, make
  // this method public.
  @VisibleForTesting
  static <K, V> SkylarkDict<K, V> copyOf(
      @Nullable Mutability mutability, Map<? extends K, ? extends V> m) {
    return SkylarkDict.<K, V>withMutability(mutability).putAllUnsafe(m);
  }

  /** @return a dict mutable in given environment only, with contents copied from given map */
  public static <K, V> SkylarkDict<K, V> copyOf(
      @Nullable Environment env, Map<? extends K, ? extends V> m) {
    return SkylarkDict.<K, V>of(env).putAllUnsafe(m);
  }

  /** Puts the given entry into the dict, without calling {@link #checkMutable}. */
  private SkylarkDict<K, V> putUnsafe(K k, V v) {
    contents.put(k, v);
    return this;
  }

  /** Puts all entries of the given map into the dict, without calling {@link #checkMutable}. */
  private <KK extends K, VV extends V> SkylarkDict<K, V> putAllUnsafe(Map<KK, VV> m) {
    for (Map.Entry<KK, VV> e : m.entrySet()) {
      contents.put(e.getKey(), e.getValue());
    }
    return this;
  }

  @Override
  public Mutability mutability() {
    return mutability;
  }

  @Override
  protected Map<K, V> getContentsUnsafe() {
    return contents;
  }

  /**
   * Puts an entry into a dict, after validating that mutation is allowed.
   *
   * @param key the key of the added entry
   * @param value the value of the added entry
   * @param loc the location to use for error reporting
   * @param mutability the {@link Mutability} associated with the opreation
   * @throws EvalException if the key is invalid or the dict is frozen
   */
  public void put(K key, V value, Location loc, Mutability mutability) throws EvalException {
    checkMutable(loc, mutability);
    EvalUtils.checkValidDictKey(key);
    contents.put(key, value);
  }

  /**
   * Convenience version of {@link #put(K, V, Location, Mutability)} that uses the {@link
   * Mutability} of an {@link Environment}.
   */
  // TODO(bazel-team): Decide whether to eliminate this overload.
  public void put(K key, V value, Location loc, Environment env) throws EvalException {
    put(key, value, loc, env.mutability());
  }

  /**
   * Puts all the entries from a given map into the dict, after validating that mutation is allowed.
   *
   * @param map the map whose entries are added
   * @param loc the location to use for error reporting
   * @param mutability the {@link Mutability} associated with the operation
   * @throws EvalException if some key is invalid or the dict is frozen
   */
  public <KK extends K, VV extends V> void putAll(
      Map<KK, VV> map, Location loc, Mutability mutability) throws EvalException {
    checkMutable(loc, mutability);
    for (Map.Entry<KK, VV> e : map.entrySet()) {
      KK k = e.getKey();
      EvalUtils.checkValidDictKey(k);
      contents.put(k, e.getValue());
    }
  }

  /**
   * Deletes the entry associated with the given key.
   *
   * @param key the key to delete
   * @param loc the location to use for error reporting
   * @param mutability the {@link Mutability} associated with the operation
   * @return the value associated to the key, or {@code null} if not present
   * @throws EvalException if the dict is frozen
   */
  V remove(Object key, Location loc, Mutability mutability) throws EvalException {
    checkMutable(loc, mutability);
    return contents.remove(key);
  }

  /**
   * Clears the dict.
   *
   * @param loc the location to use for error reporting
   * @param mutability the {@link Mutability} associated with the operation
   * @throws EvalException if the dict is frozen
   */
  void clear(Location loc, Mutability mutability) throws EvalException {
    checkMutable(loc, mutability);
    contents.clear();
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.printList(entrySet(), "{", ", ", "}", null);
  }

  @Override
  public String toString() {
    return Printer.repr(this);
  }

  /**
   * If {@code obj} is a {@code SkylarkDict}, casts it to an unmodifiable {@code Map<K, V>} after
   * checking that each of its entries has key type {@code keyType} and value type {@code
   * valueType}. If {@code obj} is {@code None} or null, treats it as an empty dict.
   *
   * <p>The returned map may or may not be a view that is affected by updates to the original dict.
   *
   * @param obj the object to cast. null and None are treated as an empty dict.
   * @param keyType the expected type of all the dict's keys
   * @param valueType the expected type of all the dict's values
   * @param description a description of the argument being converted, or null, for debugging
   */
  public static <K, V> Map<K, V> castSkylarkDictOrNoneToDict(
      Object obj, Class<K> keyType, Class<V> valueType, @Nullable String description)
      throws EvalException {
    if (EvalUtils.isNullOrNone(obj)) {
      return empty();
    }
    if (obj instanceof SkylarkDict) {
      return ((SkylarkDict<?, ?>) obj).getContents(keyType, valueType, description);
    }
    throw new EvalException(
        null,
        String.format(
            "%s is not of expected type dict or NoneType",
            description == null ? Printer.repr(obj) : String.format("'%s'", description)));
  }

  /**
   * Casts this dict to an unmodifiable {@code SkylarkDict<X, Y>}, after checking that all keys and
   * values have types {@code keyType} and {@code valueType} respectively.
   *
   * <p>The returned map may or may not be a view that is affected by updates to the original dict.
   *
   * @param keyType the expected class of keys
   * @param valueType the expected class of values
   * @param description a description of the argument being converted, or null, for debugging
   */
  @SuppressWarnings("unchecked")
  public <X, Y> Map<X, Y> getContents(
      Class<X> keyType, Class<Y> valueType, @Nullable String description)
      throws EvalException {
    Object keyDescription = description == null
        ? null : Printer.formattable("'%s' key", description);
    Object valueDescription = description == null
        ? null : Printer.formattable("'%s' value", description);
    for (Map.Entry<?, ?> e : this.entrySet()) {
      SkylarkType.checkType(e.getKey(), keyType, keyDescription);
      SkylarkType.checkType(e.getValue(), valueType, valueDescription);
    }
    return Collections.unmodifiableMap((SkylarkDict<X, Y>) this);
  }

  @Override
  public final Object getIndex(Object key, Location loc) throws EvalException {
    if (!this.containsKey(key)) {
      throw new EvalException(loc, Printer.format("key %r not found in dictionary", key));
    }
    return this.get(key);
  }

  @Override
  public final boolean containsKey(Object key, Location loc) throws EvalException {
    return this.containsKey(key);
  }

  public static <K, V> SkylarkDict<K, V> plus(
      SkylarkDict<? extends K, ? extends V> left,
      SkylarkDict<? extends K, ? extends V> right,
      @Nullable Environment env) {
    SkylarkDict<K, V> result = SkylarkDict.of(env);
    result.putAllUnsafe(left);
    result.putAllUnsafe(right);
    return result;
  }
}
