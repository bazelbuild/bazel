package net.starlark.java.eval;

import java.util.Map;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;


public interface Dict<K, V> extends Map<K, V>, StarlarkValue,
    Mutability.Freezable,
    StarlarkIndexable,
    StarlarkIterable<K> {

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
      useStarlarkThread = true)
  // TODO(adonovan): This method is named get2 as a temporary workaround for a bug in
  // StarlarkAnnotations.getStarlarkMethod. The two 'get' methods cause it to get
  // confused as to which one has the annotation. Fix it and remove "2" suffix.
  public Object get2(Object key, Object defaultValue, StarlarkThread thread) throws EvalException;



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
  public Object pop(Object key, Object defaultValue, StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "popitem",
      doc =
          "Remove and return the first <code>(key, value)</code> pair from the dictionary. "
              + "<code>popitem</code> is useful to destructively iterate over a dictionary, "
              + "as often used in set algorithms. "
              + "If the dictionary is empty, the <code>popitem</code> call fails.")
  public Tuple popitem() throws EvalException;

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
  public V setdefault(K key, V defaultValue) throws EvalException;

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
  public void update(Object pairs, Dict<String, Object> kwargs, StarlarkThread thread)
      throws EvalException;



  @StarlarkMethod(
      name = "values",
      doc =
          "Returns the list of values:"
              + "<pre class=\"language-python\">"
              + "{2: \"a\", 4: \"b\", 1: \"c\"}.values() == [\"a\", \"b\", \"c\"]</pre>\n",
      useStarlarkThread = true)
  public StarlarkList<?> values0(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "items",
      doc =
          "Returns the list of key-value tuples:"
              + "<pre class=\"language-python\">"
              + "{2: \"a\", 4: \"b\", 1: \"c\"}.items() == [(2, \"a\"), (4, \"b\"), (1, \"c\")]"
              + "</pre>\n",
      useStarlarkThread = true)
  public StarlarkList<?> items(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "keys",
      doc =
          "Returns the list of keys:"
              + "<pre class=\"language-python\">{2: \"a\", 4: \"b\", 1: \"c\"}.keys() == [2, 4, 1]"
              + "</pre>\n",
      useStarlarkThread = true)
  public StarlarkList<?> keys(StarlarkThread thread) throws EvalException;


  /**
   * Puts all the entries from a given map into the dict, after validating that mutation is allowed.
   *
   * @param map the map whose entries are added
   * @throws EvalException if some key is invalid or the dict is frozen
   */
  public <K2 extends K, V2 extends V> void putEntries(Map<K2, V2> map) throws EvalException;

  /**
   * Puts an entry into a dict, after validating that mutation is allowed.
   *
   * @param key the key of the added entry
   * @param value the value of the added entry
   * @throws EvalException if the key is invalid or the dict is frozen
   */
  public void putEntry(K key, V value) throws EvalException;

  /**
   * Deletes the entry associated with the given key.
   *
   * @param key the key to delete
   * @return the value associated to the key, or {@code null} if not present
   * @throws EvalException if the dict is frozen
   */
  V removeEntry(Object key) throws EvalException;

  /**
   * Clears the dict.
   *
   * @throws EvalException if the dict is frozen
   */
  @StarlarkMethod(name = "clear", doc = "Remove all items from the dictionary.")
  public void clearEntries() throws EvalException;

  /**
   * Casts a non-null Starlark value {@code x} to a {@code Dict<K, V>} after checking that all keys
   * and values are instances of {@code keyType} and {@code valueType}, respectively. On error, it
   * throws an EvalException whose message includes {@code what}, ideally a string literal, as a
   * description of the role of {@code x}. If x is null, it returns an immutable empty dict.
   */


  // disallowed java.util.Map update operations

  // TODO(adonovan): make mutability exception a subclass of (unchecked)
  // UnsupportedOperationException, allowing the primary Dict operations
  // to satisfy the Map operations below in the usual way (like ImmutableMap does).

  @Deprecated // use clearEntries
  @Override
  public default void clear() {
    throw new UnsupportedOperationException();
  }

  @Deprecated // use putEntry
  @Override
  public default V put(K key, V value) {
    throw new UnsupportedOperationException();
  }

  @Deprecated // use putEntries
  @Override
  public default void putAll(Map<? extends K, ? extends V> map) {
    throw new UnsupportedOperationException();
  }

  @Deprecated // use removeEntry
  @Override
  public default V remove(Object key) {
    throw new UnsupportedOperationException();
  }



}
