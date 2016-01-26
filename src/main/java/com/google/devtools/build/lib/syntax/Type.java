// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.StringCanonicalizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 *  <p>Root of Type symbol hierarchy for values in the build language.</p>
 *
 *  <p>Type symbols are primarily used for their <code>convert</code> method,
 *  which is a kind of cast operator enabling conversion from untyped (Object)
 *  references to values in the build language, to typed references.</p>
 *
 *  <p>For example, this code type-converts a value <code>x</code> returned by
 *  the evaluator, to a list of strings:</p>
 *
 *  <pre>
 *  Object x = expr.eval(env);
 *  List&lt;String&gt; s = Type.STRING_LIST.convert(x);
 *  </pre>
 */
public abstract class Type<T> {

  protected Type() {}

  /**
   * Converts untyped Object x resulting from the evaluation of an expression in the build language,
   * into a typed object of type T.
   *
   * <p>x must be *directly* convertible to this type. This therefore disqualifies "selector
   * expressions" of the form "{ config1: 'value1_of_orig_type', config2: 'value2_of_orig_type; }"
   * (which support configurable attributes). To handle those expressions, see
   * {@link com.google.devtools.build.lib.packages.BuildType#selectableConvert}.
   *
   * @param x the build-interpreter value to convert.
   * @param what a string description of what x is for; should be included in
   *    any exception thrown.  Grammatically, must describe a syntactic
   *    construct, e.g. "attribute 'srcs' of rule foo".
   * @param context the label of the current BUILD rule; must be non-null if resolution of
   *    package-relative label strings is required
   * @throws ConversionException if there was a problem performing the type conversion
   */
  public abstract T convert(Object x, String what, @Nullable Object context)
      throws ConversionException;
  // TODO(bazel-team): Check external calls (e.g. in PackageFactory), verify they always want
  // this over selectableConvert.

  /**
   * Equivalent to {@link #convert(Object, String, Object)} where the label is {@code null}.
   * Useful for converting values to types that do not involve the type {@code LABEL}
   * and hence do not require the label of the current package.
   */
  public final T convert(Object x, String what) throws ConversionException {
    return convert(x, what, null);
  }

  /**
   * Like {@link #convert(Object, String, Object)}, but converts skylark {@code None}
   * to given {@code defaultValue}.
   */
  @Nullable public final T convertOptional(Object x,
      String what, @Nullable Object context, T defaultValue)
      throws ConversionException {
    if (EvalUtils.isNullOrNone(x)) {
      return defaultValue;
    }
    return convert(x, what, context);
  }

  /**
   * Like {@link #convert(Object, String, Object)}, but converts skylark {@code None}
   * to java {@code null}.
   */
  @Nullable public final T convertOptional(Object x, String what, @Nullable Object context)
      throws ConversionException {
    return convertOptional(x, what, context, null);
  }

  /**
   * Like {@link #convert(Object, String)}, but converts skylark {@code NONE} to java {@code null}.
   */
  @Nullable public final T convertOptional(Object x, String what) throws ConversionException {
    return convertOptional(x, what, null);
  }

  public abstract T cast(Object value);

  @Override
  public abstract String toString();

  /**
   * Returns the default value for this type; may return null iff no default is defined for this
   * type.
   */
  public abstract T getDefaultValue();

  /**
   * Flatten the an instance of the type if the type is a composite one.
   *
   * <p>This is used to support reliable label visitation in
   * {@link com.google.devtools.build.lib.packages.AbstractAttributeMapper#visitLabels}. To preserve
   * that reliability, every type should faithfully define its own instance of this method. In other
   * words, be careful about defining default instances in base types that get auto-inherited by
   * their children. Keep all definitions as explicit as possible.
   */
  public abstract Collection<? extends Object> flatten(Object value);

  /**
   * {@link #flatten} return value for types that don't contain labels.
   */
  protected static final Collection<Object> NOT_COMPOSITE_TYPE = ImmutableList.of();

  /**
   * Implementation of concatenation for this type (e.g. "val1 + val2"). Returns null to
   * indicate concatenation isn't supported.
   */
  public T concat(@SuppressWarnings("unused") Iterable<T> elements) {
    return null;
  }

  /**
   * Converts an initialized Type object into a tag set representation.
   * This operation is only valid for certain sub-Types which are guaranteed
   * to be properly initialized.
   *
   * @param value the actual value
   * @throws UnsupportedOperationException if the concrete type does not support
   * tag conversion or if a convertible type has no initialized value.
   */
  public Set<String> toTagSet(Object value, String name) {
    String msg = "Attribute " + name + " does not support tag conversion.";
    throw new UnsupportedOperationException(msg);
  }

  /**
   * The type of an integer.
   */
  public static final Type<Integer> INTEGER = new IntegerType();

  /**
   * The type of a string.
   */
  public static final Type<String> STRING = new StringType();

  /**
   * The type of a boolean.
   */
  public static final Type<Boolean> BOOLEAN = new BooleanType();

  /**
   *  The type of a list of not-yet-typed objects.
   */
  public static final ObjectListType OBJECT_LIST = new ObjectListType();

  /**
   *  The type of a list of {@linkplain #STRING strings}.
   */
  public static final ListType<String> STRING_LIST = ListType.create(STRING);

  /**
   *  The type of a list of {@linkplain #INTEGER strings}.
   */
  public static final ListType<Integer> INTEGER_LIST = ListType.create(INTEGER);

  /**
   *  The type of a dictionary of {@linkplain #STRING strings}.
   */
  public static final DictType<String, String> STRING_DICT = DictType.create(STRING, STRING);

  /**
   * The type of a dictionary of {@linkplain #STRING_LIST label lists}.
   */
  public static final DictType<String, List<String>> STRING_LIST_DICT =
      DictType.create(STRING, STRING_LIST);

  /**
   * The type of a dictionary of {@linkplain #STRING strings}, where each entry
   * maps to a single string value.
   */
  public static final DictType<String, String> STRING_DICT_UNARY = DictType.create(STRING, STRING);

  /**
   *  For ListType objects, returns the type of the elements of the list; for
   *  all other types, returns null.  (This non-obvious implementation strategy
   *  is necessitated by the wildcard capture rules of the Java type system,
   *  which disallow conversion from Type{List{ELEM}} to Type{List{?}}.)
   */
  public Type<?> getListElementType() {
    return null;
  }

  /**
   *  ConversionException is thrown when a type-conversion fails; it contains
   *  an explanatory error message.
   */
  public static class ConversionException extends EvalException {
    private static String message(Type<?> type, Object value, String what) {
      StringBuilder builder = new StringBuilder();
      builder.append("expected value of type '").append(type).append("'");
      if (what != null) {
        builder.append(" for ").append(what);
      }
      builder.append(", but got ");
      Printer.write(builder, value);
      builder.append(" (").append(EvalUtils.getDataTypeName(value)).append(")");
      return builder.toString();
    }

    public ConversionException(Type<?> type, Object value, String what) {
      super(null, message(type, value, what));
    }

    public ConversionException(String message) {
      super(null, message);
    }
  }

  /********************************************************************
   *                                                                  *
   *                            Subclasses                            *
   *                                                                  *
   ********************************************************************/

  private static class ObjectType extends Type<Object> {
    @Override
    public Object cast(Object value) {
      return value;
    }

    @Override
    public String getDefaultValue() {
      throw new UnsupportedOperationException(
          "ObjectType has no default value");
    }

    @Override
    public Collection<Object> flatten(Object value) {
      return NOT_COMPOSITE_TYPE;
    }

    @Override
    public String toString() {
      return "object";
    }

    @Override
    public Object convert(Object x, String what, Object context) {
      return x;
    }
  }

  private static class IntegerType extends Type<Integer> {
    @Override
    public Integer cast(Object value) {
      return (Integer) value;
    }

    @Override
    public Integer getDefaultValue() {
      return 0;
    }

    @Override
    public Collection<Object> flatten(Object value) {
      return NOT_COMPOSITE_TYPE;
    }

    @Override
    public String toString() {
      return "int";
    }

    @Override
    public Integer convert(Object x, String what, Object context)
        throws ConversionException {
      if (!(x instanceof Integer)) {
        throw new ConversionException(this, x, what);
      }
      return (Integer) x;
    }

    @Override
    public Integer concat(Iterable<Integer> elements) {
      int ans = 0;
      for (Integer elem : elements) {
        ans += elem;
      }
      return Integer.valueOf(ans);
    }
  }

  private static class BooleanType extends Type<Boolean> {
    @Override
    public Boolean cast(Object value) {
      return (Boolean) value;
    }

    @Override
    public Boolean getDefaultValue() {
      return false;
    }

    @Override
    public Collection<Object> flatten(Object value) {
      return NOT_COMPOSITE_TYPE;
    }

    @Override
    public String toString() {
      return "boolean";
    }

    // Conversion to boolean must also tolerate integers of 0 and 1 only.
    @Override
    public Boolean convert(Object x, String what, Object context)
        throws ConversionException {
      if (x instanceof Boolean) {
        return (Boolean) x;
      }
      Integer xAsInteger = INTEGER.convert(x, what, context);
      if (xAsInteger == 0) {
        return false;
      } else if (xAsInteger == 1) {
        return true;
      }
      throw new ConversionException("boolean is not one of [0, 1]");
    }

    /**
     * Booleans attributes are converted to tags based on their names.
     */
    @Override
    public Set<String> toTagSet(Object value, String name) {
      if (value == null) {
        String msg = "Illegal tag conversion from null on Attribute " + name  + ".";
        throw new IllegalStateException(msg);
      }
      String tag = (Boolean) value ? name : "no" + name;
      return ImmutableSet.of(tag);
    }
  }

  private static class StringType extends Type<String> {
    @Override
    public String cast(Object value) {
      return (String) value;
    }

    @Override
    public String getDefaultValue() {
      return "";
    }

    @Override
    public Collection<Object> flatten(Object value) {
      return NOT_COMPOSITE_TYPE;
    }

    @Override
    public String toString() {
      return "string";
    }

    @Override
    public String convert(Object x, String what, Object context)
        throws ConversionException {
      if (!(x instanceof String)) {
        throw new ConversionException(this, x, what);
      }
      return StringCanonicalizer.intern((String) x);
    }

    @Override
    public String concat(Iterable<String> elements) {
      return Joiner.on("").join(elements);
    }

    /**
     * A String is representable as a set containing its value.
     */
    @Override
    public Set<String> toTagSet(Object value, String name) {
      if (value == null) {
        String msg = "Illegal tag conversion from null on Attribute " + name + ".";
        throw new IllegalStateException(msg);
      }
      return ImmutableSet.of((String) value);
    }
  }

  /**
   * A type to support dictionary attributes.
   */
  public static class DictType<KeyT, ValueT> extends Type<Map<KeyT, ValueT>> {

    private final Type<KeyT> keyType;
    private final Type<ValueT> valueType;

    private final Map<KeyT, ValueT> empty = ImmutableMap.of();

    public static <KEY, VALUE> DictType<KEY, VALUE> create(
        Type<KEY> keyType, Type<VALUE> valueType) {
      return new DictType<>(keyType, valueType);
    }

    private DictType(Type<KeyT> keyType, Type<ValueT> valueType) {
      this.keyType = keyType;
      this.valueType = valueType;
    }

    public Type<KeyT> getKeyType() {
      return keyType;
    }

    public Type<ValueT> getValueType() {
      return valueType;
    }

    @SuppressWarnings("unchecked")
    @Override
    public Map<KeyT, ValueT> cast(Object value) {
      return (Map<KeyT, ValueT>) value;
    }

    @Override
    public String toString() {
      return "dict(" + keyType + ", " + valueType + ")";
    }

    @Override
    public Map<KeyT, ValueT> convert(Object x, String what, Object context)
        throws ConversionException {
      if (!(x instanceof Map<?, ?>)) {
        throw new ConversionException(String.format(
            "Expected a map for dictionary but got a %s", x.getClass().getName())); 
      }
      // Order the keys so the return value will be independent of insertion order.
      Map<KeyT, ValueT> result = new TreeMap<>();
      Map<?, ?> o = (Map<?, ?>) x;
      for (Entry<?, ?> elem : o.entrySet()) {
        result.put(
            keyType.convert(elem.getKey(), "dict key element", context),
            valueType.convert(elem.getValue(), "dict value element", context));
      }
      return ImmutableMap.copyOf(result);
    }

    @Override
    public Map<KeyT, ValueT> getDefaultValue() {
      return empty;
    }

    @Override
    public Collection<Object> flatten(Object value) {
      ImmutableList.Builder<Object> result = ImmutableList.builder();
      for (Map.Entry<KeyT, ValueT> entry : cast(value).entrySet()) {
        result.addAll(keyType.flatten(entry.getKey()));
        result.addAll(valueType.flatten(entry.getValue()));
      }
      return result.build();
    }
  }

  /** A type for lists of a given element type */
  public static class ListType<ElemT> extends Type<List<ElemT>> {

    private final Type<ElemT> elemType;

    private final List<ElemT> empty = ImmutableList.of();

    public static <ELEM> ListType<ELEM> create(Type<ELEM> elemType) {
      return new ListType<>(elemType);
    }

    private ListType(Type<ElemT> elemType) {
      this.elemType = elemType;
    }

    @SuppressWarnings("unchecked")
    @Override
    public List<ElemT> cast(Object value) {
      return (List<ElemT>) value;
    }

    @Override
    public Type<ElemT> getListElementType() {
      return elemType;
    }

    @Override
    public List<ElemT> getDefaultValue() {
      return empty;
    }

    @Override
    public Collection<Object> flatten(Object value) {
      ImmutableList.Builder<Object> labels = ImmutableList.builder();
      for (ElemT entry : cast(value)) {
        labels.addAll(elemType.flatten(entry));
      }
      return labels.build();
    }

    @Override
    public String toString() {
      return "list(" + elemType + ")";
    }

    @Override
    public List<ElemT> convert(Object x, String what, Object context)
        throws ConversionException {
      if (!(x instanceof Iterable<?>)) {
        throw new ConversionException(this, x, what);
      }
      int index = 0;
      Iterable<?> iterable = (Iterable<?>) x;
      List<ElemT> result = new ArrayList<>(Iterables.size(iterable));
      for (Object elem : iterable) {
        ElemT converted = elemType.convert(elem, "element " + index + " of " + what, context);
        if (converted != null) {
          result.add(converted);
        } else {
          // shouldn't happen but it does, rarely
          String message = "Converting a list with a null element: "
              + "element " + index + " of " + what + " in " + context;
          LoggingUtil.logToRemote(Level.WARNING, message,
              new ConversionException(message));
        }
        ++index;
      }
      // We preserve GlobList-s so they can make it to attributes;
      // some external code relies on attributes preserving this information.
      // TODO(bazel-team): somehow make Skylark extensible enough that
      // GlobList support can be wholly moved out of Skylark into an extension.
      if (x instanceof GlobList<?>) {
        return new GlobList<>(((GlobList<?>) x).getCriteria(), result);
      }
      if (x instanceof MutableList) {
        GlobList<?> globList = ((MutableList) x).getGlobList();
        if (globList != null) {
          return new GlobList<>(globList.getCriteria(), result);
        }
      }
      return result;
    }

    @Override
    public List<ElemT> concat(Iterable<List<ElemT>> elements) {
      ImmutableList.Builder<ElemT> builder = ImmutableList.builder();
      for (List<ElemT> list : elements) {
        builder.addAll(list);
      }
      return builder.build();
    }

    /**
     * A list is representable as a tag set as the contents of itself expressed
     * as Strings. So a {@code List<String>} is effectively converted to a {@code Set<String>}.
     */
    @Override
    public Set<String> toTagSet(Object items, String name) {
      if (items == null) {
        String msg = "Illegal tag conversion from null on Attribute" + name + ".";
        throw new IllegalStateException(msg);
      }
      Set<String> tags = new LinkedHashSet<>();
      @SuppressWarnings("unchecked")
      List<ElemT> itemsAsListofElem = (List<ElemT>) items;
      for (ElemT element : itemsAsListofElem) {
        tags.add(element.toString());
      }
      return tags;
    }
  }

  /** Type for lists of arbitrary objects */
  public static class ObjectListType extends ListType<Object> {

    private static final Type<Object> elemType = new ObjectType();

    private ObjectListType() {
      super(elemType);
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<Object> convert(Object x, String what, Object context)
        throws ConversionException {
      if (x instanceof SkylarkList) {
        return ((SkylarkList) x).getImmutableList();
      } else if (x instanceof List) {
        return (List<Object>) x;
      } else if (x instanceof Iterable) {
        // Do not remove <Object>: workaround for Java 7 type inference.
        return ImmutableList.<Object>copyOf((Iterable<?>) x);
      } else {
        throw new ConversionException(this, x, what);
      }
    }
  }

  /**
   * The type of a general list.
   */
  public static final ListType<Object> LIST = new ListType<>(new ObjectType());
}
