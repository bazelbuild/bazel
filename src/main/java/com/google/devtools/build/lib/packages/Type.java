// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.License.LicenseParsingException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.SelectorValue;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.StringCanonicalizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
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

  private Type() {}

  /**
   * Converts untyped Object x resulting from the evaluation of an expression in the build language,
   * into a typed object of type T.
   *
   * <p>x must be *directly* convertible to this type. This therefore disqualifies "selector
   * expressions" of the form "{ config1: 'value1_of_orig_type', config2: 'value2_of_orig_type; }"
   * (which support configurable attributes). To handle those expressions, see
   * {@link #selectableConvert}.
   *
   * @param x the build-interpreter value to convert.
   * @param what a string description of what x is for; should be included in
   *    any exception thrown.  Grammatically, must describe a syntactic
   *    construct, e.g. "attribute 'srcs' of rule foo".
   * @param currentRule the label of the current BUILD rule; must be non-null if resolution of
   *    package-relative label strings is required
   * @throws ConversionException if there was a problem performing the type conversion
   */
  public abstract T convert(Object x, String what, @Nullable Label currentRule)
      throws ConversionException;
  // TODO(bazel-team): Check external calls (e.g. in PackageFactory), verify they always want
  // this over selectableConvert.

  /**
   * Equivalent to {@link #convert(Object, String, Label)} where the label is {@code null}.
   * Useful for converting values to types that do not involve the type {@code LABEL}
   * and hence do not require the label of the current package.
   */
  public final T convert(Object x, String what) throws ConversionException {
    return convert(x, what, null);
  }

  /**
   * Like {@link #convert(Object, String, Label)}, but converts skylark {@code None}
   * to given {@code defaultValue}.
   */
  @Nullable public final T convertOptional(Object x,
      String what, @Nullable Label currentRule, T defaultValue)
      throws ConversionException {
    if (EvalUtils.isNullOrNone(x)) {
      return defaultValue;
    }
    return convert(x, what, currentRule);
  }

  /**
   * Like {@link #convert(Object, String, Label)}, but converts skylark {@code None}
   * to java {@code null}.
   */
  @Nullable public final T convertOptional(Object x, String what, @Nullable Label currentRule)
      throws ConversionException {
    return convertOptional(x, what, currentRule, null);
  }

  /**
   * Like {@link #convert(Object, String)}, but converts skylark {@code NONE} to java {@code null}.
   */
  @Nullable public final T convertOptional(Object x, String what) throws ConversionException {
    return convertOptional(x, what, null);
  }

  /**
   * Variation of {@link #convert} that supports selector expressions for configurable attributes
   * (i.e. "{ config1: 'value1_of_orig_type', config2: 'value2_of_orig_type; }"). If x is a
   * selector expression, returns a {@link Selector} instance that contains key-mapped entries
   * of the native type. Else, returns the native type directly.
   *
   * <p>The caller is responsible for casting the returned value appropriately.
   */
  public Object selectableConvert(Object x, String what, @Nullable Label currentRule)
      throws ConversionException {
    if (x instanceof com.google.devtools.build.lib.syntax.SelectorList) {
      return new SelectorList<T>(
          ((com.google.devtools.build.lib.syntax.SelectorList) x).getElements(),
          what, currentRule, this);
    } else {
      return convert(x, what, currentRule);
    }
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
   * If this type contains labels (e.g. it *is* a label or it's a collection of labels),
   * returns a list of those labels for a value of that type. If this type doesn't
   * contain labels, returns an empty list.
   *
   * <p>This is used to support reliable label visitation in
   * {@link AbstractAttributeMapper#visitLabels}. To preserve that reliability, every
   * type should faithfully define its own instance of this method. In other words,
   * be careful about defining default instances in base types that get auto-inherited
   * by their children. Keep all definitions as explicit as possible.
   */
  public abstract Collection<Label> getLabels(Object value);

  /**
   * {@link #getLabels} return value for types that don't contain labels.
   */
  private static final Collection<Label> NO_LABELS_HERE = ImmutableList.of();

  /**
   * Implementation of concatenation for this type (e.g. "val1 + val2"). Returns null to
   * designate concatenation isn't supported.
   */
  public T concat(Iterable<T> elements) {
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
   * The type of a TriState with values: true (x>0), false (x==0), auto (x<0).
   */
  public static final Type<TriState> TRISTATE = new TriStateType();

  /**
   * The type of a label. Labels are not actually a first-class datatype in
   * the build language, but they are so frequently used in the definitions of
   * attributes that it's worth treating them specially (and providing support
   * for resolution of relative-labels in the <code>convert()</code> method).
   */
  public static final Type<Label> LABEL = new LabelType();

  /**
   * This is a label type that does not cause dependencies. It is needed because
   * certain rules want to verify the type of a target referenced by one of their attributes, but
   * if there was a dependency edge there, it would be a circular dependency.
   */
  public static final Type<Label> NODEP_LABEL = new LabelType();

  /**
   * The type of a license. Like Label, licenses aren't first-class, but
   * they're important enough to justify early syntax error detection.
   */
  public static final Type<License> LICENSE = new LicenseType();

  /**
   * The type of a single distribution.  Only used internally, as a type
   * symbol, not a converter.
   */
  public static final Type<DistributionType> DISTRIBUTION = new Type<DistributionType>() {
    @Override
    public DistributionType cast(Object value) {
      return (DistributionType) value;
    }

    @Override
    public DistributionType convert(Object x, String what, Label currentRule) {
      throw new UnsupportedOperationException();
    }

    @Override
    public DistributionType getDefaultValue() {
      return null;
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "distribution";
    }
  };

  /**
   * The type of a set of distributions. Distributions are not a first-class type,
   * but they do warrant early syntax checking.
   */
  public static final Type<Set<DistributionType>> DISTRIBUTIONS = new Distributions();

  /**
   *  The type of an output file, treated as a {@link #LABEL}.
   */
  public static final Type<Label> OUTPUT = new OutputType();

  /**
   * The type of a FilesetEntry attribute inside a Fileset.
   */
  public static final Type<FilesetEntry> FILESET_ENTRY = new FilesetEntryType();

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
   *  The type of a list of {@linkplain #OUTPUT outputs}.
   */
  public static final ListType<Label> OUTPUT_LIST = ListType.create(OUTPUT);

  /**
   *  The type of a list of {@linkplain #LABEL labels}.
   */
  public static final ListType<Label> LABEL_LIST = ListType.create(LABEL);

  /**
   *  The type of a list of {@linkplain #NODEP_LABEL labels} that do not cause
   *  dependencies.
   */
  public static final ListType<Label> NODEP_LABEL_LIST = ListType.create(NODEP_LABEL);

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
   * The type of a dictionary of {@linkplain #LABEL_LIST label lists}.
   */
  public static final DictType<String, List<Label>> LABEL_LIST_DICT =
      DictType.create(STRING, LABEL_LIST);

  /**
   * The type of a dictionary of {@linkplain #LABEL labels}.
   */
  public static final DictType<String, Label> LABEL_DICT_UNARY = DictType.create(STRING, LABEL);

  /**
   * The type of a list of {@linkplain #FILESET_ENTRY FilesetEntries}.
   */
  public static final ListType<FilesetEntry> FILESET_ENTRY_LIST = ListType.create(FILESET_ENTRY);

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

    private ConversionException(Type<?> type, Object value, String what) {
      super(null, message(type, value, what));
    }

    private ConversionException(String message) {
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
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "object";
    }

    @Override
    public Object convert(Object x, String what, Label currentRule) {
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
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "int";
    }

    @Override
    public Integer convert(Object x, String what, Label currentRule)
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
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "boolean";
    }

    // Conversion to boolean must also tolerate integers of 0 and 1 only.
    @Override
    public Boolean convert(Object x, String what, Label currentRule)
        throws ConversionException {
      if (x instanceof Boolean) {
        return (Boolean) x;
      }
      Integer xAsInteger = INTEGER.convert(x, what, currentRule);
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

  /**
   * Tristate values are needed for cases where user intent matters.
   *
   * <p>Tristate values are not explicitly interchangeable with booleans and are
   * handled explicitly as TriStates. Prefer Booleans with default values where
   * possible.  The main use case for TriState values is when a Rule's behavior
   * must interact with a Flag value in a complicated way.</p>
   */
  private static class TriStateType extends Type<TriState> {
    @Override
    public TriState cast(Object value) {
      return (TriState) value;
    }

    @Override
    public TriState getDefaultValue() {
      return TriState.AUTO;
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "tristate";
    }

    // Like BooleanType, this must handle integers as well.
    @Override
    public TriState convert(Object x, String what, Label currentRule)
        throws ConversionException {
      if (x instanceof TriState) {
        return (TriState) x;
      }
      if (x instanceof Boolean) {
        return ((Boolean) x) ? TriState.YES : TriState.NO;
      }
      Integer xAsInteger = INTEGER.convert(x, what, currentRule);
      if (xAsInteger == -1) {
        return TriState.AUTO;
      } else if (xAsInteger == 1) {
        return TriState.YES;
      } else if (xAsInteger == 0) {
        return TriState.NO;
      }
      throw new ConversionException(this, x, "TriState values is not one of [-1, 0, 1]");
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
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "string";
    }

    @Override
    public String convert(Object x, String what, Label currentRule)
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

  private static class FilesetEntryType extends Type<FilesetEntry> {
    @Override
    public FilesetEntry cast(Object value) {
      return (FilesetEntry) value;
    }

    @Override
    public FilesetEntry convert(Object x, String what, Label currentRule)
        throws ConversionException {
      if (!(x instanceof FilesetEntry)) {
        throw new ConversionException(this, x, what);
      }
      return (FilesetEntry) x;
    }

    @Override
    public String toString() {
      return "FilesetEntry";
    }

    @Override
    public FilesetEntry getDefaultValue() {
      return null;
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      return cast(value).getLabels();
    }
  }

  private static class LabelType extends Type<Label> {
    @Override
    public Label cast(Object value) {
      return (Label) value;
    }

    @Override
    public Label getDefaultValue() {
      return null; // Labels have no default value
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      return ImmutableList.of(cast(value));
    }

    @Override
    public String toString() {
      return "label";
    }

    @Override
    public Label convert(Object x, String what, Label currentRule)
        throws ConversionException {
      if (x instanceof Label) {
        return (Label) x;
      }
      try {
        return currentRule.getRelative(
            STRING.convert(x, what, currentRule));
      } catch (Label.SyntaxException e) {
        throw new ConversionException("invalid label '" + x + "' in "
            + what + ": " + e.getMessage());
      }
    }
  }

  /**
   * Like Label, LicenseType is a derived type, which is declared specially
   * in order to allow syntax validation. It represents the licenses, as
   * described in {@ref License}.
   */
  public static class LicenseType extends Type<License> {
    @Override
    public License cast(Object value) {
      return (License) value;
    }

    @Override
    public License convert(Object x, String what, Label currentRule) throws ConversionException {
      try {
        List<String> licenseStrings = STRING_LIST.convert(x, what);
        return License.parseLicense(licenseStrings);
      } catch (LicenseParsingException e) {
        throw new ConversionException(e.getMessage());
      }
    }

    @Override
    public License getDefaultValue() {
      return License.NO_LICENSE;
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "license";
    }
  }

  /**
   * Like Label, Distributions is a derived type, which is declared specially
   * in order to allow syntax validation. It represents the declared distributions
   * of a target, as described in {@ref License}.
   */
  private static class Distributions extends Type<Set<DistributionType>> {
    @SuppressWarnings("unchecked")
    @Override
    public Set<DistributionType> cast(Object value) {
      return (Set<DistributionType>) value;
    }

    @Override
    public Set<DistributionType> convert(Object x, String what, Label currentRule)
        throws ConversionException {
      try {
        List<String> distribStrings = STRING_LIST.convert(x, what);
        return License.parseDistributions(distribStrings);
      } catch (LicenseParsingException e) {
        throw new ConversionException(e.getMessage());
      }
    }

    @Override
    public Set<DistributionType> getDefaultValue() {
      return Collections.emptySet();
    }

    @Override
    public Collection<Label> getLabels(Object what) {
      return NO_LABELS_HERE;
    }

    @Override
    public String toString() {
      return "distributions";
    }

    @Override
    public Type<DistributionType> getListElementType() {
      return DISTRIBUTION;
    }
  }

  private static class OutputType extends Type<Label> {
    @Override
    public Label cast(Object value) {
      return (Label) value;
    }

    @Override
    public Label getDefaultValue() {
      return null;
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      return ImmutableList.of(cast(value));
    }

    @Override
    public String toString() {
      return "output";
    }

    @Override
    public Label convert(Object x, String what, Label currentRule)
        throws ConversionException {

      String value;
      try {
        value = STRING.convert(x, what, currentRule);
      } catch (ConversionException e) {
        throw new ConversionException(this, x, what);
      }
      try {
        // Enforce value is relative to the currentRule.
        Label result = currentRule.getRelative(value);
        if (!result.getPackageName().equals(currentRule.getPackageName())) {
          throw new ConversionException("label '" + value + "' is not in the current package");
        }
        return result;
      } catch (Label.SyntaxException e) {
        throw new ConversionException(
            "illegal output file name '" + value + "' in rule " + currentRule + ": "
            + e.getMessage());
      }
    }
  }

  /**
   * A type to support dictionary attributes.
   */
  public static class DictType<KeyT, ValueT> extends Type<Map<KeyT, ValueT>> {

    private final Type<KeyT> keyType;
    private final Type<ValueT> valueType;

    private final Map<KeyT, ValueT> empty = ImmutableMap.of();

    private static <KEY, VALUE> DictType<KEY, VALUE> create(
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
    public Map<KeyT, ValueT> convert(Object x, String what, Label currentRule)
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
            keyType.convert(elem.getKey(), "dict key element", currentRule),
            valueType.convert(elem.getValue(), "dict value element", currentRule));
      }
      return ImmutableMap.copyOf(result);
    }

    @Override
    public Map<KeyT, ValueT> getDefaultValue() {
      return empty;
    }

    @Override
    public Collection<Label> getLabels(Object value) {
      ImmutableList.Builder<Label> labels = ImmutableList.builder();
      for (Map.Entry<KeyT, ValueT> entry : cast(value).entrySet()) {
        labels.addAll(keyType.getLabels(entry.getKey()));
        labels.addAll(valueType.getLabels(entry.getValue()));
      }
      return labels.build();
    }
  }

  /** A type for lists of a given element type */
  public static class ListType<ElemT> extends Type<List<ElemT>> {

    private final Type<ElemT> elemType;

    private final List<ElemT> empty = ImmutableList.of();

    private static <ELEM> ListType<ELEM> create(Type<ELEM> elemType) {
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
    public Collection<Label> getLabels(Object value) {
      ImmutableList.Builder<Label> labels = ImmutableList.builder();
      for (ElemT entry : cast(value)) {
        labels.addAll(elemType.getLabels(entry));
      }
      return labels.build();
    }

    @Override
    public String toString() {
      return "list(" + elemType + ")";
    }

    @Override
    public List<ElemT> convert(Object x, String what, Label currentRule)
        throws ConversionException {
      if (!(x instanceof Iterable<?>)) {
        throw new ConversionException(this, x, what);
      }
      List<ElemT> result = new ArrayList<>();
      int index = 0;
      for (Object elem : (Iterable<?>) x) {
        ElemT converted = elemType.convert(elem, "element " + index + " of " + what, currentRule);
        if (converted != null) {
          result.add(converted);
        } else {
          // shouldn't happen but it does, rarely
          String message = "Converting a list with a null element: "
              + "element " + index + " of " + what + " in " + currentRule;
          LoggingUtil.logToRemote(Level.WARNING, message,
              new ConversionException(message));
        }
        ++index;
      }
      if (x instanceof GlobList<?>) {
        return new GlobList<>(((GlobList<?>) x).getCriteria(), result);
      } else {
        return result;
      }
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
    public List<Object> convert(Object x, String what, Label currentRule)
        throws ConversionException {
      if (x instanceof SkylarkList) {
        return ((SkylarkList) x).toList();
      } else if (x instanceof List) {
        return (List<Object>) x;
      } else if (x instanceof Iterable) {
        return ImmutableList.copyOf((Iterable<?>) x);
      } else {
        throw new ConversionException(this, x, what);
      }
    }
  }

  /**
   * The type of a general list.
   */
  public static final ListType<Object> LIST = new ListType<>(new ObjectType());

  /**
   * Returns whether the specified type is a label type or not.
   */
  public static boolean isLabelType(Type<?> type) {
    return type == LABEL || type == LABEL_LIST || type == LABEL_DICT_UNARY
        || type == NODEP_LABEL || type == NODEP_LABEL_LIST
        || type == LABEL_LIST_DICT || type == FILESET_ENTRY_LIST;
  }

  /**
   * Special Type that represents a selector expression for configurable attributes. Holds a
   * mapping of {@code <Label, T>} entries, where keys are configurability patterns and values are
   * objects of the attribute's native Type.
   */
  public static final class Selector<T> {
    private final Type<T> originalType;
    private final Map<Label, T> map;
    private final Label defaultConditionLabel;
    private final boolean hasDefaultCondition;

    /**
     * Value to use when none of an attribute's selection criteria match.
     */
    @VisibleForTesting
    public static final String DEFAULT_CONDITION_KEY = "//conditions:default";

    @VisibleForTesting
    Selector(Object x, String what, @Nullable Label currentRule, Type<T> originalType)
        throws ConversionException {
      Preconditions.checkState(x instanceof Map<?, ?>);

      try {
        defaultConditionLabel = Label.parseAbsolute(DEFAULT_CONDITION_KEY);
      } catch (Label.SyntaxException e) {
        throw new IllegalStateException(DEFAULT_CONDITION_KEY + " is not a valid label");
      }

      this.originalType = originalType;
      Map<Label, T> result = Maps.newLinkedHashMap();
      boolean foundDefaultCondition = false;
      for (Entry<?, ?> entry : ((Map<?, ?>) x).entrySet()) {
        Label key = LABEL.convert(entry.getKey(), what, currentRule);
        if (key.equals(defaultConditionLabel)) {
          foundDefaultCondition = true;
        }
        result.put(key, originalType.convert(entry.getValue(), what, currentRule));
      }
      map = ImmutableMap.copyOf(result);
      hasDefaultCondition = foundDefaultCondition;
    }

    /**
     * Returns the selector's (configurability pattern --gt; matching values) map.
     */
    public Map<Label, T> getEntries() {
      return map;
    }

    /**
     * Returns the value to use when none of the attribute's selection keys match.
     */
    public T getDefault() {
      return map.get(defaultConditionLabel);
    }

    /**
     * Returns whether or not this selector has a default condition.
     */
    public boolean hasDefault() {
      return hasDefaultCondition;
    }

    /**
     * Returns the native Type for this attribute (i.e. what this would be if it wasn't a
     * selector expression).
     */
    public Type<T> getOriginalType() {
      return originalType;
    }

    /**
     * Returns true for labels that are "reserved selector key words" and not intended to
     * map to actual targets.
     */
    public static boolean isReservedLabel(Label label) {
      return label.toString().equals(DEFAULT_CONDITION_KEY);
    }
  }

  /**
   * Holds an ordered collection of {@link Selector}s. This is used to support
   * {@code attr = rawValue + select(...) + select(...) + ..."} syntax. For consistency's
   * sake, raw values are stored as selects with only a default condition.
   */
  public static final class SelectorList<T> {
    private final Type<T> originalType;
    private final List<Selector<T>> elements;

    @VisibleForTesting
    SelectorList(List<Object> x, String what, @Nullable Label currentRule,
        Type<T> originalType) throws ConversionException {
      if (x.size() > 1 && originalType.concat(ImmutableList.<T>of()) == null) {
        throw new ConversionException(
            String.format("type '%s' doesn't support select concatenation", originalType));
      }

      ImmutableList.Builder<Selector<T>> builder = ImmutableList.builder();
      for (Object elem : x) {
        if (elem instanceof SelectorValue) {
          builder.add(new Selector<T>(((SelectorValue) elem).getDictionary(), what,
              currentRule, originalType));
        } else {
          T directValue = originalType.convert(elem, what, currentRule);
          builder.add(new Selector<T>(ImmutableMap.of(Selector.DEFAULT_CONDITION_KEY, directValue),
              what, currentRule, originalType));
        }
      }
      this.originalType = originalType;
      this.elements = builder.build();
    }

    /**
     * Returns a syntactically order-preserved list of all values and selectors for this attribute.
     */
    public List<Selector<T>> getSelectors() {
      return elements;
    }

    /**
     * Returns the native Type for this attribute (i.e. what this would be if it wasn't a
     * selector list).
     */
    public Type<T> getOriginalType() {
      return originalType;
    }

    /**
     * Returns the labels of all configurability keys across all selects in this expression.
     */
    public Set<Label> getKeyLabels() {
      ImmutableSet.Builder<Label> keys = ImmutableSet.builder();
      for (Selector<T> selector : getSelectors()) {
         for (Label label : selector.getEntries().keySet()) {
           if (!Selector.isReservedLabel(label)) {
             keys.add(label);
           }
         }
      }
      return keys.build();
    }
  }
}
