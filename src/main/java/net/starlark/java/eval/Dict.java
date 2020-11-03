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

package net.starlark.java.eval;

import static java.lang.Math.max;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/**
 * A Dict is a Starlark dictionary (dict), a mapping from keys to values.
 *
 * <p>Dicts are iterable in both Java and Starlark; the iterator yields successive keys.
 *
 * <p>Starlark operations on dicts, including element update {@code dict[k]=v} and the {@code
 * update} and {@code setdefault} methods, may insert arbitrary Starlark values as dict keys/values,
 * regardless of the type argument used to reference the dict from Java code. Therefore, as long as
 * a dict is mutable, Java code should refer to it only through a type such as {@code Dict<Object,
 * Object>} or {@code Dict<?, ?>} to avoid undermining the type-safety of the Java application. Once
 * the dict becomes frozen, it is safe to {@link #cast} it to a more specific type that accurately
 * reflects its entries, such as {@code Dict<String, StarlarkInt>}.
 *
 * <p>The following Dict methods, defined by the {@link Map} interface, are not supported. Use the
 * corresponding methods with "entry" in their name; they may report mutation failure by throwing a
 * checked exception:
 *
 * <pre>
 * void clear()         -- use clearEntries
 * V put(K, V)          -- use putEntry
 * void putAll(Map)     -- use putEntries
 * V remove(Object key) -- use removeEntry
 * </pre>
 */
@StarlarkBuiltin(
    name = "dict",
    category = "core",
    doc =
        "dict is a built-in type representing an associative mapping or <i>dictionary</i>. A"
            + " dictionary supports indexing using <code>d[k]</code> and key membership testing"
            + " using <code>k in d</code>; both operations take constant time. Unfrozen"
            + " dictionaries are mutable, and may be updated by assigning to <code>d[k]</code> or"
            + " by calling certain methods. Dictionaries are iterable; iteration yields the"
            + " sequence of keys in insertion order. Iteration order is unaffected by updating the"
            + " value associated with an existing key, but is affected by removing then"
            + " reinserting a key.\n"
            + "<pre>d = {0: 0, 2: 2, 1: 1}\n"
            + "[k for k in d]  # [0, 2, 1]\n"
            + "d.pop(2)\n"
            + "d[0], d[2] = \"a\", \"b\"\n"
            + "0 in d, \"a\" in d  # (True, False)\n"
            + "[(k, v) for k, v in d.items()]  # [(0, \"a\"), (1, 1), (2, \"b\")]\n"
            + "</pre>\n"
            + "<p>There are three ways to construct a dictionary:\n"
            + "<ol>\n"
            + "<li>A dictionary expression <code>{k: v, ...}</code> yields a new dictionary with"
            + " the specified key/value entries, inserted in the order they appear in the"
            + " expression. Evaluation fails if any two key expressions yield the same"
            + " value.\n"
            + "<li>A dictionary comprehension <code>{k: v for vars in seq}</code> yields a new"
            + " dictionary into which each key/value pair is inserted in loop iteration order."
            + " Duplicates are permitted: the first insertion of a given key determines its"
            + " position in the sequence, and the last determines its associated value.\n"
            + "<pre class=\"language-python\">\n"
            + "{k: v for k, v in ((\"a\", 0), (\"b\", 1), (\"a\", 2))}  # {\"a\": 2, \"b\": 1}\n"
            + "{i: 2*i for i in range(3)}  # {0: 0, 1: 2, 2: 4}\n"
            + "</pre>\n"
            + "<li>A call to the built-in <a href=\"globals.html#dict\">dict</a> function returns"
            + " a dictionary containing the specified entries, which are inserted in argument"
            + " order, positional arguments before named. As with comprehensions, duplicate keys"
            + " are permitted.\n"
            + "</ol>")
public final class Dict<K, V>
    implements Map<K, V>,
        StarlarkValue,
        Mutability.Freezable,
        StarlarkIndexable,
        StarlarkIterable<K> {

  private final LinkedHashMap<K, V> contents;
  private int iteratorCount; // number of active iterators (unused once frozen)

  /** Final except for {@link #unsafeShallowFreeze}; must not be modified any other way. */
  private Mutability mutability;

  private Dict(@Nullable Mutability mutability, LinkedHashMap<K, V> contents) {
    this.mutability = mutability == null ? Mutability.IMMUTABLE : mutability;
    this.contents = contents;
  }

  private Dict(@Nullable Mutability mutability) {
    this(mutability, new LinkedHashMap<>());
  }

  private Dict(@Nullable Mutability mutability, int initialCapacity) {
    this(mutability, new LinkedHashMap<>(initialCapacity));
  }

  /**
   * Takes ownership of the supplied LinkedHashMap and returns a new Dict that wraps it. The caller
   * must not subsequently modify the map, but the Dict may do so.
   */
  static <K, V> Dict<K, V> wrap(@Nullable Mutability mutability, LinkedHashMap<K, V> contents) {
    return new Dict<>(mutability, contents);
  }

  @Override
  public boolean truth() {
    return !isEmpty();
  }

  @Override
  public boolean isImmutable() {
    return mutability().isFrozen();
  }

  @Override
  public boolean updateIteratorCount(int delta) {
    if (mutability().isFrozen()) {
      return false;
    }
    if (delta > 0) {
      iteratorCount++;
    } else if (delta < 0) {
      iteratorCount--;
    }
    return iteratorCount > 0;
  }

  @Override
  public void checkHashable() throws EvalException {
    // Even a frozen dict is unhashable.
    throw Starlark.errorf("unhashable type: 'dict'");
  }

  @Override
  public int hashCode() {
    return contents.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    return contents.equals(o);
  }

  @Override
  public Iterator<K> iterator() {
    return contents.keySet().iterator();
  }

  @StarlarkMethod(
      name = "get",
      doc =
          "Returns the value for <code>key</code> if <code>key</code> is in the dictionary, "
              + "else <code>default</code>. If <code>default</code> is not given, it defaults to "
              + "<code>None</code>, so that this method never throws an error.",
      parameters = {
        @Param(name = "key", doc = "The key to look for."),
        @Param(
            name = "default",
            defaultValue = "None",
            named = true,
            doc = "The default value to use (instead of None) if the key is not found.")
      },
      allowReturnNones = true,
      useStarlarkThread = true)
  // TODO(adonovan): This method is named get2 as a temporary workaround for a bug in
  // StarlarkAnnotations.getStarlarkMethod. The two 'get' methods cause it to get
  // confused as to which one has the annotation. Fix it and remove "2" suffix.
  public Object get2(Object key, Object defaultValue, StarlarkThread thread) throws EvalException {
    Object v = this.get(key);
    if (v != null) {
      return v;
    }

    // This statement is executed for its effect, which is to throw "unhashable"
    // if key is unhashable, instead of returning defaultValue.
    // I think this is a bug: the correct behavior is simply 'return defaultValue'.
    // See https://github.com/bazelbuild/starlark/issues/65.
    containsKey(thread.getSemantics(), key);

    return defaultValue;
  }

  @StarlarkMethod(
      name = "pop",
      doc =
          "Removes a <code>key</code> from the dict, and returns the associated value. "
              + "If no entry with that key was found, remove nothing and return the specified "
              + "<code>default</code> value; if no default value was specified, fail instead.",
      parameters = {
        @Param(name = "key", doc = "The key."),
        @Param(
            name = "default",
            defaultValue = "unbound",
            named = true,
            doc = "a default value if the key is absent."),
      },
      useStarlarkThread = true)
  public Object pop(Object key, Object defaultValue, StarlarkThread thread) throws EvalException {
    Object value = get(key);
    if (value != null) {
      removeEntry(key);
      return value;
    }
    if (defaultValue != Starlark.UNBOUND) {
      return defaultValue;
    }
    // TODO(adonovan): improve error; this ain't Python.
    throw Starlark.errorf("KeyError: %s", Starlark.repr(key));
  }

  @StarlarkMethod(
      name = "popitem",
      doc =
          "Remove and return an arbitrary <code>(key, value)</code> pair from the dictionary. "
              + "<code>popitem()</code> is useful to destructively iterate over a dictionary, "
              + "as often used in set algorithms. "
              + "If the dictionary is empty, calling <code>popitem()</code> fails. "
              + "It is deterministic which pair is returned.",
      useStarlarkThread = true)
  public Tuple popitem(StarlarkThread thread) throws EvalException {
    if (isEmpty()) {
      throw Starlark.errorf("popitem(): dictionary is empty");
    }
    Object key = keySet().iterator().next();
    Object value = get(key);
    removeEntry(key);
    return Tuple.pair(key, value);
  }

  @StarlarkMethod(
      name = "setdefault",
      doc =
          "If <code>key</code> is in the dictionary, return its value. "
              + "If not, insert key with a value of <code>default</code> "
              + "and return <code>default</code>. "
              + "<code>default</code> defaults to <code>None</code>.",
      parameters = {
        @Param(name = "key", doc = "The key."),
        @Param(
            name = "default",
            defaultValue = "None",
            named = true,
            doc = "a default value if the key is absent."),
      })
  @SuppressWarnings("unchecked") // Cast of value to V
  public Object setdefault(K key, Object defaultValue) throws EvalException {
    // TODO(adonovan): opt: use putIfAbsent to avoid hashing twice.
    Object value = get(key);
    if (value != null) {
      return value;
    }
    putEntry(key, (V) defaultValue);
    return defaultValue;
  }

  @StarlarkMethod(
      name = "update",
      doc =
          "Updates the dictionary first with the optional positional argument, <code>pairs</code>, "
              + " then with the optional keyword arguments\n"
              + "If the positional argument is present, it must be a dict, iterable, or None.\n"
              + "If it is a dict, then its key/value pairs are inserted into this dict. "
              + "If it is an iterable, it must provide a sequence of pairs (or other iterables "
              + "of length 2), each of which is treated as a key/value pair to be inserted.\n"
              + "Each keyword argument <code>name=value</code> causes the name/value "
              + "pair to be inserted into this dict.",
      parameters = {
        @Param(
            name = "pairs",
            defaultValue = "[]",
            doc =
                "Either a dictionary or a list of entries. Entries must be tuples or lists with "
                    + "exactly two elements: key, value."),
      },
      extraKeywords = @Param(name = "kwargs", doc = "Dictionary of additional entries."),
      useStarlarkThread = true)
  @SuppressWarnings("unchecked")
  public NoneType update(Object args, Dict<String, Object> kwargs, StarlarkThread thread)
      throws EvalException {
    // TODO(adonovan): opt: don't materialize dict; call put directly.

    // These types and casts are unsafe; see class doc comment.
    Dict<K, V> dict =
        args instanceof Dict
            ? (Dict<K, V>) args
            : getDictFromArgs("update", args, thread.mutability());
    dict = Dict.plus(dict, (Dict<K, V>) kwargs, thread.mutability());
    putEntries(dict);
    return Starlark.NONE;
  }

  @StarlarkMethod(
      name = "values",
      doc =
          "Returns the list of values:"
              + "<pre class=\"language-python\">"
              + "{2: \"a\", 4: \"b\", 1: \"c\"}.values() == [\"a\", \"b\", \"c\"]</pre>\n",
      useStarlarkThread = true)
  public StarlarkList<?> values0(StarlarkThread thread) throws EvalException {
    return StarlarkList.copyOf(thread.mutability(), values());
  }

  @StarlarkMethod(
      name = "items",
      doc =
          "Returns the list of key-value tuples:"
              + "<pre class=\"language-python\">"
              + "{2: \"a\", 4: \"b\", 1: \"c\"}.items() == [(2, \"a\"), (4, \"b\"), (1, \"c\")]"
              + "</pre>\n",
      useStarlarkThread = true)
  public StarlarkList<?> items(StarlarkThread thread) throws EvalException {
    Object[] array = new Object[size()];
    int i = 0;
    for (Map.Entry<?, ?> e : entrySet()) {
      array[i++] = Tuple.pair(e.getKey(), e.getValue());
    }
    return StarlarkList.wrap(thread.mutability(), array);
  }

  @StarlarkMethod(
      name = "keys",
      doc =
          "Returns the list of keys:"
              + "<pre class=\"language-python\">{2: \"a\", 4: \"b\", 1: \"c\"}.keys() == [2, 4, 1]"
              + "</pre>\n",
      useStarlarkThread = true)
  public StarlarkList<?> keys(StarlarkThread thread) throws EvalException {
    Object[] array = new Object[size()];
    int i = 0;
    for (Map.Entry<?, ?> e : entrySet()) {
      array[i++] = e.getKey();
    }
    return StarlarkList.wrap(thread.mutability(), array);
  }

  private static final Dict<?, ?> EMPTY = of(Mutability.IMMUTABLE);

  /** Returns an immutable empty dict. */
  // Safe because the empty singleton is immutable.
  @SuppressWarnings("unchecked")
  public static <K, V> Dict<K, V> empty() {
    return (Dict<K, V>) EMPTY;
  }

  /** Returns a new empty dict with the specified mutability. */
  public static <K, V> Dict<K, V> of(@Nullable Mutability mu) {
    return new Dict<>(mu);
  }

  /** Returns a new dict with the specified mutability and a single entry. */
  public static <K, V> Dict<K, V> of(@Nullable Mutability mu, K k, V v) {
    return new Dict<K, V>(mu).putUnsafe(k, v);
  }

  /** Returns a new dict with the specified mutability and two entries. */
  public static <K, V> Dict<K, V> of(@Nullable Mutability mu, K k1, V v1, K k2, V v2) {
    return new Dict<K, V>(mu).putUnsafe(k1, v1).putUnsafe(k2, v2);
  }

  /** Returns a new dict with the specified mutability containing the entries of {@code m}. */
  public static <K, V> Dict<K, V> copyOf(@Nullable Mutability mu, Map<? extends K, ? extends V> m) {
    if (mu == null && m instanceof Dict && ((Dict) m).isImmutable()) {
      @SuppressWarnings("unchecked")
      Dict<K, V> dict = (Dict<K, V>) m; // safe
      return dict;
    }

    Dict<K, V> dict = new Dict<>(mu);
    for (Map.Entry<? extends K, ? extends V> e : m.entrySet()) {
      dict.contents.put(
          Starlark.checkValid(e.getKey()), //
          Starlark.checkValid(e.getValue()));
    }
    return dict;
  }

  /** Returns an immutable dict containing the entries of {@code m}. */
  public static <K, V> Dict<K, V> immutableCopyOf(Map<? extends K, ? extends V> m) {
    return copyOf(null, m);
  }

  /** Returns a new empty Dict.Builder. */
  public static <K, V> Builder<K, V> builder() {
    return new Builder<>();
  }

  /** A reusable builder for Dicts. */
  public static final class Builder<K, V> {
    private final ArrayList<Object> items = new ArrayList<>(); // [k, v, ... k, v]

    /** Adds an entry (k, v) to the builder, overwriting any previous entry with the same key . */
    public Builder<K, V> put(K k, V v) {
      items.add(Starlark.checkValid(k));
      items.add(Starlark.checkValid(v));
      return this;
    }

    /** Adds all the map's entries to the builder. */
    public Builder<K, V> putAll(Map<? extends K, ? extends V> map) {
      items.ensureCapacity(items.size() + 2 * map.size());
      for (Map.Entry<? extends K, ? extends V> e : map.entrySet()) {
        put(e.getKey(), e.getValue());
      }
      return this;
    }

    /** Returns a new immutable Dict containing the entries added so far. */
    public Dict<K, V> buildImmutable() {
      return build(null);
    }

    /**
     * Returns a new Dict containing the entries added so far. The result has the specified
     * mutability; null means immutable.
     */
    public Dict<K, V> build(@Nullable Mutability mu) {
      int n = items.size() / 2;
      LinkedHashMap<K, V> map = Maps.newLinkedHashMapWithExpectedSize(n);
      for (int i = 0; i < n; i++) {
        @SuppressWarnings("unchecked")
        K k = (K) items.get(2 * i); // safe
        @SuppressWarnings("unchecked")
        V v = (V) items.get(2 * i + 1); // safe
        map.put(k, v);
      }
      return wrap(mu, map);
    }
  }

  /** Puts the given entry into the dict, without calling {@link #checkMutable}. */
  private Dict<K, V> putUnsafe(K k, V v) {
    contents.put(k, v);
    return this;
  }

  @Override
  public Mutability mutability() {
    return mutability;
  }

  @Override
  public void unsafeShallowFreeze() {
    Mutability.Freezable.checkUnsafeShallowFreezePrecondition(this);
    this.mutability = Mutability.IMMUTABLE;
  }

  /**
   * Puts an entry into a dict, after validating that mutation is allowed.
   *
   * @param key the key of the added entry
   * @param value the value of the added entry
   * @throws EvalException if the key is invalid or the dict is frozen
   */
  public void putEntry(K key, V value) throws EvalException {
    Starlark.checkMutable(this);
    Starlark.checkHashable(key);
    contents.put(key, value);
  }

  /**
   * Puts all the entries from a given map into the dict, after validating that mutation is allowed.
   *
   * @param map the map whose entries are added
   * @throws EvalException if some key is invalid or the dict is frozen
   */
  public <K2 extends K, V2 extends V> void putEntries(Map<K2, V2> map) throws EvalException {
    Starlark.checkMutable(this);
    for (Map.Entry<K2, V2> e : map.entrySet()) {
      K2 k = e.getKey();
      Starlark.checkHashable(k);
      contents.put(k, e.getValue());
    }
  }

  /**
   * Deletes the entry associated with the given key.
   *
   * @param key the key to delete
   * @return the value associated to the key, or {@code null} if not present
   * @throws EvalException if the dict is frozen
   */
  V removeEntry(Object key) throws EvalException {
    Starlark.checkMutable(this);
    return contents.remove(key);
  }

  /**
   * Clears the dict.
   *
   * @throws EvalException if the dict is frozen
   */
  @StarlarkMethod(name = "clear", doc = "Remove all items from the dictionary.")
  public void clearEntries() throws EvalException {
    Starlark.checkMutable(this);
    contents.clear();
  }

  @Override
  public void repr(Printer printer) {
    printer.printList(entrySet(), "{", ", ", "}");
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }

  /**
   * Casts a non-null Starlark value {@code x} to a {@code Dict<K, V>} after checking that all keys
   * and values are instances of {@code keyType} and {@code valueType}, respectively. On error, it
   * throws an EvalException whose message includes {@code what}, ideally a string literal, as a
   * description of the role of {@code x}. If x is null, it returns an immutable empty dict.
   */
  public static <K, V> Dict<K, V> cast(Object x, Class<K> keyType, Class<V> valueType, String what)
      throws EvalException {
    Preconditions.checkNotNull(x);
    if (!(x instanceof Dict)) {
      throw Starlark.errorf("got %s for '%s', want dict", Starlark.type(x), what);
    }

    for (Map.Entry<?, ?> e : ((Map<?, ?>) x).entrySet()) {
      if (!keyType.isAssignableFrom(e.getKey().getClass())
          || !valueType.isAssignableFrom(e.getValue().getClass())) {
        // TODO(adonovan): change message to "found <K2, V2> entry",
        // without suggesting that the entire dict is <K2, V2>.
        throw Starlark.errorf(
            "got dict<%s, %s> for '%s', want dict<%s, %s>",
            Starlark.type(e.getKey()),
            Starlark.type(e.getValue()),
            what,
            Starlark.classType(keyType),
            Starlark.classType(valueType));
      }
    }

    @SuppressWarnings("unchecked") // safe
    Dict<K, V> res = (Dict<K, V>) x;
    return res;
  }

  /** Like {@link #cast}, but if x is None, returns an empty Dict. */
  public static <K, V> Dict<K, V> noneableCast(
      Object x, Class<K> keyType, Class<V> valueType, String what) throws EvalException {
    return x == Starlark.NONE ? empty() : cast(x, keyType, valueType, what);
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    Object v = get(key);
    if (v == null) {
      throw Starlark.errorf("key %s not found in dictionary", Starlark.repr(key));
    }
    return v;
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    Starlark.checkHashable(key);
    return this.containsKey(key);
  }

  static <K, V> Dict<K, V> plus(
      Dict<? extends K, ? extends V> left,
      Dict<? extends K, ? extends V> right,
      @Nullable Mutability mu) {
    Dict<K, V> result = new Dict<>(mu, max(left.size(), right.size()));
    // Update underlying map contents directly, input dicts already contain valid objects
    result.contents.putAll(left.contents);
    result.contents.putAll(right.contents);
    return result;
  }

  @SuppressWarnings("unchecked")
  static <K, V> Dict<K, V> getDictFromArgs(String funcname, Object args, @Nullable Mutability mu)
      throws EvalException {
    Iterable<?> seq;
    try {
      seq = Starlark.toIterable(args);
    } catch (EvalException ex) {
      throw Starlark.errorf("in %s, got %s, want iterable", funcname, Starlark.type(args));
    }
    Dict<K, V> result = Dict.of(mu);
    int pos = 0;
    for (Object item : seq) {
      Iterable<?> seq2;
      try {
        seq2 = Starlark.toIterable(item);
      } catch (EvalException ex) {
        throw Starlark.errorf(
            "in %s, dictionary update sequence element #%d is not iterable (%s)",
            funcname, pos, Starlark.type(item));
      }
      // TODO(adonovan): opt: avoid unnecessary allocations and copies.
      // Why is there no operator to compute len(x), following the spec, without iterating??
      List<Object> pair = Lists.newArrayList(seq2);
      if (pair.size() != 2) {
        throw Starlark.errorf(
            "in %s, item #%d has length %d, but exactly two elements are required",
            funcname, pos, pair.size());
      }
      // These casts are lies
      result.putEntry((K) pair.get(0), (V) pair.get(1));
      pos++;
    }
    return result;
  }

  // java.util.Map accessors

  @Override
  public boolean containsKey(Object key) {
    return contents.containsKey(key);
  }

  @Override
  public boolean containsValue(Object value) {
    return contents.containsValue(value);
  }

  @Override
  public Set<Map.Entry<K, V>> entrySet() {
    return Collections.unmodifiableMap(contents).entrySet();
  }

  @Override
  public V get(Object key) {
    return contents.get(key);
  }

  @Override
  public boolean isEmpty() {
    return contents.isEmpty();
  }

  @Override
  public Set<K> keySet() {
    return Collections.unmodifiableMap(contents).keySet();
  }

  @Override
  public int size() {
    return contents.size();
  }

  @Override
  public Collection<V> values() {
    return Collections.unmodifiableMap(contents).values();
  }

  // disallowed java.util.Map update operations

  // TODO(adonovan): make mutability exception a subclass of (unchecked)
  // UnsupportedOperationException, allowing the primary Dict operations
  // to satisfy the Map operations below in the usual way (like ImmutableMap does).

  @Deprecated // use clearEntries
  @Override
  public void clear() {
    throw new UnsupportedOperationException();
  }

  @Deprecated // use putEntry
  @Override
  public V put(K key, V value) {
    throw new UnsupportedOperationException();
  }

  @Deprecated // use putEntries
  @Override
  public void putAll(Map<? extends K, ? extends V> map) {
    throw new UnsupportedOperationException();
  }

  @Deprecated // use removeEntry
  @Override
  public V remove(Object key) {
    throw new UnsupportedOperationException();
  }
}
