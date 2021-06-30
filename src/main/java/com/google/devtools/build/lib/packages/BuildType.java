// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.License.LicenseParsingException;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.packages.Type.DictType;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.packages.Type.ListType;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkValue;

/**
 * Collection of data types that are specific to building things, i.e. not inherent to Starlark.
 *
 * <p>BEFORE YOU ADD A NEW TYPE: See javadoc in {@link Type}.
 */
public final class BuildType {

  /**
   * The type of a label. Labels are not actually a first-class datatype in the build language, but
   * they are so frequently used in the definitions of attributes that it's worth treating them
   * specially (and providing support for resolution of relative-labels in the <code>convert()
   * </code> method).
   */
  @AutoCodec public static final Type<Label> LABEL = new LabelType(LabelClass.DEPENDENCY);
  /** The type of a dictionary of {@linkplain #LABEL labels}. */
  @AutoCodec
  public static final DictType<String, Label> LABEL_DICT_UNARY =
      DictType.create(Type.STRING, LABEL);
  /** The type of a dictionary keyed by {@linkplain #LABEL labels} with string values. */
  @AutoCodec
  public static final DictType<Label, String> LABEL_KEYED_STRING_DICT =
      LabelKeyedDictType.create(Type.STRING);
  /** The type of a list of {@linkplain #LABEL labels}. */
  @AutoCodec public static final ListType<Label> LABEL_LIST = ListType.create(LABEL);
  /**
   * This is a label type that does not cause dependencies. It is needed because certain rules want
   * to verify the type of a target referenced by one of their attributes, but if there was a
   * dependency edge there, it would be a circular dependency.
   */
  @AutoCodec
  public static final Type<Label> NODEP_LABEL = new LabelType(LabelClass.NONDEP_REFERENCE);
  /** The type of a list of {@linkplain #NODEP_LABEL labels} that do not cause dependencies. */
  @AutoCodec public static final ListType<Label> NODEP_LABEL_LIST = ListType.create(NODEP_LABEL);
  /**
   * The type of a license. Like Label, licenses aren't first-class, but they're important enough to
   * justify early syntax error detection.
   */
  @AutoCodec public static final Type<License> LICENSE = new LicenseType();
  /** The type of a single distribution. Only used internally, as a type symbol, not a converter. */
  @AutoCodec
  static final Type<DistributionType> DISTRIBUTION =
      new Type<DistributionType>() {
        @Override
        public DistributionType cast(Object value) {
          return (DistributionType) value;
        }

        @Override
        public DistributionType convert(Object x, Object what, Object context) {
          throw new UnsupportedOperationException();
        }

        @Override
        public DistributionType getDefaultValue() {
          return null;
        }

        @Override
        public void visitLabels(
            LabelVisitor visitor, DistributionType value, @Nullable Attribute context) {}

        @Override
        public String toString() {
          return "distribution";
        }
      };
  /**
   * The type of a set of distributions. Distributions are not a first-class type, but they do
   * warrant early syntax checking.
   */
  @AutoCodec public static final Type<Set<DistributionType>> DISTRIBUTIONS = new Distributions();
  /** The type of an output file, treated as a {@link #LABEL}. */
  @AutoCodec public static final Type<Label> OUTPUT = new OutputType();
  /** The type of a list of {@linkplain #OUTPUT outputs}. */
  @AutoCodec public static final ListType<Label> OUTPUT_LIST = ListType.create(OUTPUT);
  /** The type of a FilesetEntry attribute inside a Fileset. */
  @AutoCodec public static final Type<FilesetEntry> FILESET_ENTRY = new FilesetEntryType();
  /** The type of a list of {@linkplain #FILESET_ENTRY FilesetEntries}. */
  @AutoCodec
  public static final ListType<FilesetEntry> FILESET_ENTRY_LIST = ListType.create(FILESET_ENTRY);
  /** The type of a TriState with values: true (x>0), false (x==0), auto (x<0). */
  @AutoCodec public static final Type<TriState> TRISTATE = new TriStateType();

  private BuildType() {
    // Do not instantiate
  }

  /**
   * Returns whether the specified type is a label type or not.
   */
  public static boolean isLabelType(Type<?> type) {
    return type.getLabelClass() != LabelClass.NONE;
  }

  /**
   * Variation of {@link Type#convert} that supports selector expressions for configurable
   * attributes* (i.e. "{ config1: 'value1_of_orig_type', config2: 'value2_of_orig_type; }"). If x
   * is a selector expression, returns a {@link Selector} instance that contains key-mapped entries
   * of the native type. Else, returns the native type directly.
   *
   * <p>The caller is responsible for casting the returned value appropriately.
   */
  public static <T> Object selectableConvert(
      Type<T> type, Object x, Object what, LabelConversionContext context)
      throws ConversionException {
    if (x instanceof com.google.devtools.build.lib.packages.SelectorList) {
      return new SelectorList<>(
          ((com.google.devtools.build.lib.packages.SelectorList) x).getElements(),
          what,
          context,
          type);
    } else {
      return type.convert(x, what, context);
    }
  }

  private static final class FilesetEntryType extends Type<FilesetEntry> {
    @Override
    public FilesetEntry cast(Object value) {
      return (FilesetEntry) value;
    }

    @Override
    public FilesetEntry convert(Object x, Object what, Object context)
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
    public LabelClass getLabelClass() {
      return LabelClass.FILESET_ENTRY;
    }

    @Override
    public FilesetEntry getDefaultValue() {
      return null;
    }

    @Override
    public void visitLabels(LabelVisitor visitor, FilesetEntry value, @Nullable Attribute context) {
      for (Label label : value.getLabels()) {
        visitor.visit(label, context);
      }
    }
  }

  /** Context in which to evaluate a label with repository remappings */
  public static class LabelConversionContext {
    private final Label label;
    private final ImmutableMap<RepositoryName, RepositoryName> repositoryMapping;
    private final HashMap<String, Label> convertedLabelsInPackage;

    public LabelConversionContext(
        Label label,
        ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
        HashMap<String, Label> convertedLabelsInPackage) {
      this.label = label;
      this.repositoryMapping = repositoryMapping;
      this.convertedLabelsInPackage = convertedLabelsInPackage;
    }

    Label getLabel() {
      return label;
    }

    /** Returns the Label corresponding to the input, using the current conversion context. */
    public Label convert(String input) throws LabelSyntaxException {
      // Optimization: First check the package-local map, avoiding Label validation, Label
      // construction, and global Interner lookup. This approach tends to be very profitable
      // overall, since it's common for the targets in a single package to have duplicate
      // label-strings across all their attribute values.
      Label converted = convertedLabelsInPackage.get(input);
      if (converted == null) {
        converted = label.getRelativeWithRemapping(input, repositoryMapping);
        convertedLabelsInPackage.put(input, converted);
      }
      return converted;
    }

    ImmutableMap<RepositoryName, RepositoryName> getRepositoryMapping() {
      return repositoryMapping;
    }

    HashMap<String, Label> getConvertedLabelsInPackage() {
      return convertedLabelsInPackage;
    }

    @Override
    public String toString() {
      return label.toString();
    }
  }

  private static class LabelType extends Type<Label> {
    private final LabelClass labelClass;

    LabelType(LabelClass labelClass) {
      this.labelClass = labelClass;
    }

    @Override
    public Label cast(Object value) {
      return (Label) value;
    }

    @Override
    public Label getDefaultValue() {
      return null; // Labels have no default value
    }

    @Override
    public void visitLabels(LabelVisitor visitor, Label value, @Nullable Attribute context) {
      visitor.visit(value, context);
    }

    @Override
    public String toString() {
      return "label";
    }

    @Override
    public LabelClass getLabelClass() {
      return labelClass;
    }

    @Override
    public Label convert(Object x, Object what, Object context)
        throws ConversionException {
      if (x instanceof Label) {
        return (Label) x;
      }
      try {
        if (!(x instanceof String)) {
          throw new ConversionException(Type.STRING, x, what);
        }
        // This String here is about to be parsed into a Label. We do not use STRING.convert since
        // there is absolutely no motivation to intern the String; the Label we create will be
        // storing a reference to different string (a substring in fact).
        String str = (String) x;
        // TODO(b/110101445): check if context is ever actually null
        if (context == null) {
          return Label.parseAbsolute(
              str, /* defaultToMain= */ false, /* repositoryMapping= */ ImmutableMap.of());
          // TODO(b/110308446): remove instances of context being a Label
        } else if (context instanceof Label) {
          return ((Label) context).getRelativeWithRemapping(str, ImmutableMap.of());
        } else if (context instanceof LabelConversionContext) {
          LabelConversionContext labelConversionContext = (LabelConversionContext) context;
          return labelConversionContext.convert(str);
        } else {
          throw new ConversionException("invalid context '" + context + "' in " + what);
        }
      } catch (LabelSyntaxException e) {
        throw new ConversionException("invalid label '" + x + "' in "
            + what + ": " + e.getMessage());
      }
    }
  }

  /**
   * Dictionary type specialized for label keys, which is able to detect collisions caused by the
   * fact that labels have multiple equivalent representations in Starlark code.
   */
  private static class LabelKeyedDictType<ValueT> extends DictType<Label, ValueT> {
    private LabelKeyedDictType(Type<ValueT> valueType) {
      super(LABEL, valueType, LabelClass.DEPENDENCY);
    }

    static <ValueT> LabelKeyedDictType<ValueT> create(Type<ValueT> valueType) {
      Preconditions.checkArgument(
          valueType.getLabelClass() == LabelClass.NONE
              || valueType.getLabelClass() == LabelClass.DEPENDENCY,
          "Values associated with label keys must not be labels themselves.");
      return new LabelKeyedDictType<>(valueType);
    }

    @Override
    public Map<Label, ValueT> convert(Object x, Object what, Object context)
        throws ConversionException {
      Map<Label, ValueT> result = super.convert(x, what, context);
      // The input is known to be a map because super.convert succeeded; otherwise, a
      // ConversionException would have been thrown.
      Map<?, ?> input = (Map<?, ?>) x;

      if (input.size() == result.size()) {
        // No collisions found. Exit early.
        return result;
      }
      // Look for collisions in order to produce a nicer error message.
      Map<Label, List<Object>> convertedFrom = new LinkedHashMap<>();
      for (Object original : input.keySet()) {
        Label label = LABEL.convert(original, what, context);
        convertedFrom.computeIfAbsent(label, k -> new ArrayList<>()).add(original);
      }
      Printer errorMessage = new Printer();
      errorMessage.append("duplicate labels");
      if (what != null) {
        errorMessage.append(" in ").append(what.toString());
      }
      errorMessage.append(':');
      boolean isFirstEntry = true;
      for (Map.Entry<Label, List<Object>> entry : convertedFrom.entrySet()) {
        if (entry.getValue().size() == 1) {
          continue;
        }
        if (isFirstEntry) {
          isFirstEntry = false;
        } else {
          errorMessage.append(',');
        }
        errorMessage.append(' ');
        errorMessage.str(entry.getKey());
        errorMessage.append(" (as ");
        errorMessage.repr(entry.getValue());
        errorMessage.append(')');
      }
      throw new ConversionException(errorMessage.toString());
    }
  }

  /**
   * Like Label, LicenseType is a derived type, which is declared specially in order to allow syntax
   * validation. It represents the licenses, as described in {@link License}.
   */
  public static final class LicenseType extends Type<License> {
    @Override
    public License cast(Object value) {
      return (License) value;
    }

    @Override
    public License convert(Object x, Object what, Object context) throws ConversionException {
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
    public void visitLabels(LabelVisitor visitor, License value, @Nullable Attribute context) {}

    @Override
    public String toString() {
      return "license";
    }
  }

  /**
   * Like Label, Distributions is a derived type, which is declared specially in order to allow
   * syntax validation. It represents the declared distributions of a target, as described in {@link
   * License}.
   */
  private static final class Distributions extends Type<Set<DistributionType>> {
    @SuppressWarnings("unchecked")
    @Override
    public Set<DistributionType> cast(Object value) {
      return (Set<DistributionType>) value;
    }

    @Override
    public Set<DistributionType> convert(Object x, Object what, Object context)
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
    public void visitLabels(
        LabelVisitor visitor, Set<DistributionType> value, @Nullable Attribute context) {}

    @Override
    public String toString() {
      return "distributions";
    }

    @Override
    public Type<DistributionType> getListElementType() {
      return DISTRIBUTION;
    }
  }

  private static final class OutputType extends Type<Label> {
    @Override
    public Label cast(Object value) {
      return (Label) value;
    }

    @Override
    public Label getDefaultValue() {
      return null;
    }

    @Override
    public void visitLabels(LabelVisitor visitor, Label value, @Nullable Attribute context) {
      visitor.visit(value, context);
    }

    @Override
    public LabelClass getLabelClass() {
      return LabelClass.OUTPUT;
    }

    @Override
    public String toString() {
      return "output";
    }

    @Override
    public Label convert(Object x, Object what, Object context)
        throws ConversionException {

      String value;
      try {
        value = STRING.convert(x, what, context);
      } catch (ConversionException e) {
        throw new ConversionException(this, x, what);
      }
      try {
        // Enforce value is relative to the context.
        Label currentRule;
        ImmutableMap<RepositoryName, RepositoryName> repositoryMapping;
        if (context instanceof LabelConversionContext) {
          currentRule = ((LabelConversionContext) context).getLabel();
          repositoryMapping = ((LabelConversionContext) context).getRepositoryMapping();
        } else {
          throw new ConversionException("invalid context '" + context + "' in " + what);
        }
        Label result = currentRule.getRelativeWithRemapping(value, repositoryMapping);
        if (!result.getPackageIdentifier().equals(currentRule.getPackageIdentifier())) {
          throw new ConversionException("label '" + value + "' is not in the current package");
        }
        return result;
      } catch (LabelSyntaxException e) {
        throw new ConversionException(
            "illegal output file name '" + value + "' in rule " + context + ": " + e.getMessage());
      }
    }
  }

  /**
   * Holds an ordered collection of {@link Selector}s. This is used to support {@code attr =
   * rawValue + select(...) + select(...) + ..."} syntax. For consistency's sake, raw values are
   * stored as selects with only a default condition.
   */
  // TODO(adonovan): merge with packages.Selector{List,Value}.
  // We don't need three classes for the same concept.
  public static final class SelectorList<T> implements StarlarkValue {
    private final Type<T> originalType;
    private final List<Selector<T>> elements;

    @VisibleForTesting
    SelectorList(
        List<Object> x, Object what, @Nullable LabelConversionContext context, Type<T> originalType)
        throws ConversionException {
      if (x.size() > 1 && originalType.concat(ImmutableList.of()) == null) {
        throw new ConversionException(
            String.format("type '%s' doesn't support select concatenation", originalType));
      }

      ImmutableList.Builder<Selector<T>> builder = ImmutableList.builder();
      for (Object elem : x) {
        if (elem instanceof SelectorValue) {
          builder.add(new Selector<>(((SelectorValue) elem).getDictionary(), what,
              context, originalType, ((SelectorValue) elem).getNoMatchError()));
        } else {
          T directValue = originalType.convert(elem, what, context);
          builder.add(new Selector<>(ImmutableMap.of(Selector.DEFAULT_CONDITION_KEY, directValue),
              what, context, originalType));
        }
      }
      this.originalType = originalType;
      this.elements = builder.build();
    }

    SelectorList(List<Selector<T>> elements, Type<T> originalType) {
      this.elements = ImmutableList.copyOf(elements);
      this.originalType = originalType;
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
      for (Selector<T> selector : elements) {
        for (Label label : selector.getEntries().keySet()) {
          if (!Selector.isReservedLabel(label)) {
            keys.add(label);
          }
        }
      }
      return keys.build();
    }

    @Override
    public String toString() {
      return Starlark.repr(this);
    }

    @Override
    public void repr(Printer printer) {
      // Convert to a lib.packages.SelectorList to guarantee consistency with callers that serialize
      // directly on that type.
      List<SelectorValue> selectorValueList = new ArrayList<>();
      for (Selector<T> element : elements) {
        selectorValueList.add(new SelectorValue(element.getEntries(), element.getNoMatchError()));
      }
      try {
        printer.repr(com.google.devtools.build.lib.packages.SelectorList.of(selectorValueList));
      } catch (EvalException e) {
        throw new IllegalStateException("this list should have been validated on creation");
      }
    }
  }

  /** Lazy string message to pass as the {@code what} when converting a select branch value. */
  private static final class SelectBranchMessage {
    private final Object what;
    private final Label key;

    SelectBranchMessage(Object what, Label key) {
      this.what = what;
      this.key = key;
    }

    @Override
    public String toString() {
      return String.format("each branch in select expression of %s (including '%s')", what, key);
    }
  }

  /**
   * Special Type that represents a selector expression for configurable attributes. Holds a
   * mapping of {@code <Label, T>} entries, where keys are configurability patterns and values are
   * objects of the attribute's native Type.
   */
  public static final class Selector<T> {
    /** Value to use when none of an attribute's selection criteria match. */
    @VisibleForTesting
    public static final String DEFAULT_CONDITION_KEY = "//conditions:default";

    static final Label DEFAULT_CONDITION_LABEL =
        Label.parseAbsoluteUnchecked(DEFAULT_CONDITION_KEY);

    private final Type<T> originalType;
    // Can hold null values, underlying implementation should be ordered.
    private final Map<Label, T> map;
    private final Set<Label> conditionsWithDefaultValues;
    private final String noMatchError;
    private final boolean hasDefaultCondition;

    /** Creates a new Selector using the default error message when no conditions match. */
    Selector(
        ImmutableMap<?, ?> x,
        Object what,
        @Nullable LabelConversionContext context,
        Type<T> originalType)
        throws ConversionException {
      this(x, what, context, originalType, "");
    }

    /** Creates a new Selector with a custom error message for when no conditions match. */
    Selector(
        ImmutableMap<?, ?> x,
        Object what,
        @Nullable LabelConversionContext context,
        Type<T> originalType,
        String noMatchError)
        throws ConversionException {
      this.originalType = originalType;
      LinkedHashMap<Label, T> result = Maps.newLinkedHashMapWithExpectedSize(x.size());
      ImmutableSet.Builder<Label> defaultValuesBuilder = ImmutableSet.builder();
      boolean foundDefaultCondition = false;
      for (Map.Entry<?, ?> entry : x.entrySet()) {
        Label key = LABEL.convert(entry.getKey(), what, context);
        if (key.equals(DEFAULT_CONDITION_LABEL)) {
          foundDefaultCondition = true;
        }
        if (entry.getValue() == Starlark.NONE) {
          // { "//condition": None } is the same as not setting the value.
          result.put(key, originalType.getDefaultValue());
          defaultValuesBuilder.add(key);
        } else {
          Object selectBranch = what == null ? null : new SelectBranchMessage(what, key);
          result.put(key, originalType.convert(entry.getValue(), selectBranch, context));
        }
      }
      this.map = Collections.unmodifiableMap(result);
      this.noMatchError = noMatchError;
      this.conditionsWithDefaultValues = defaultValuesBuilder.build();
      this.hasDefaultCondition = foundDefaultCondition;
    }

    /**
     * Create a new Selector from raw values. A defensive copy of the supplied map is <i>not</i>
     * made, so it imperative that it is not modified following construction.
     */
    Selector(
        LinkedHashMap<Label, T> map,
        Type<T> originalType,
        String noMatchError,
        ImmutableSet<Label> conditionsWithDefaultValues,
        boolean hasDefaultCondition) {
      this.originalType = originalType;
      this.map = Collections.unmodifiableMap(map);
      this.noMatchError = noMatchError;
      this.conditionsWithDefaultValues = conditionsWithDefaultValues;
      this.hasDefaultCondition = hasDefaultCondition;
    }

    /**
     * Returns the selector's (configurability pattern --gt; matching values) map.
     *
     * <p>Entries in this map retain the order of the entries in the map provided to the {@link
     * #Selector} constructor.
     */
    public Map<Label, T> getEntries() {
      return map;
    }

    /**
     * Returns the value to use when none of the attribute's selection keys match.
     */
    public T getDefault() {
      return map.get(DEFAULT_CONDITION_LABEL);
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
     * Returns true if this selector has the structure: {"//conditions:default": ...}. That means
     * all values are always chosen.
     */
    public boolean isUnconditional() {
      return map.size() == 1 && hasDefaultCondition;
    }

    /**
     * Returns true if an explicit value is set for the given condition, vs. { "//condition": None }
     * which means revert to the default.
     */
    public boolean isValueSet(Label condition) {
      return !conditionsWithDefaultValues.contains(condition);
    }

    /**
     * Returns a custom error message for this select when no condition matches, or an empty
     * string if no such message is declared.
     */
    public String getNoMatchError() {
      return noMatchError;
    }

    /**
     * Returns true for labels that are "reserved selector key words" and not intended to
     * map to actual targets.
     */
    public static boolean isReservedLabel(Label label) {
      return DEFAULT_CONDITION_LABEL.equals(label);
    }
  }

  /**
   * A TriState value is like a boolean attribute whose default value may be distinguished from
   * either of the possible explicitly assigned values. TriState attributes may be assigned the
   * values 0 (NO), 1 (YES), or None (AUTO). TriState is deprecated; use attr.int(values=[-1, 0, 1])
   * instead.
   */
  private static final class TriStateType extends Type<TriState> {
    @Override
    public TriState cast(Object value) {
      return (TriState) value;
    }

    @Override
    public TriState getDefaultValue() {
      return TriState.AUTO;
    }

    @Override
    public void visitLabels(LabelVisitor visitor, TriState value, @Nullable Attribute context) {}

    @Override
    public String toString() {
      return "tristate";
    }

    @Override
    public TriState convert(Object x, Object what, Object context)
        throws ConversionException {
      if (x instanceof TriState) {
        return (TriState) x;
      }
      if (x instanceof Boolean) {
        // TODO(adonovan): re-enable this under flag control; see b/116691720.
        // throw new ConversionException(this, x,
        //   "rule attribute (tristate is being replaced by "
        //       + "attr.int(values=[-1, 0, 1]), and it no longer accepts Boolean values; "
        //       + "instead, use 0 or 1, or None for the default)");
        return ((Boolean) x) ? TriState.YES : TriState.NO;
      }
      int xAsInteger = INTEGER.convert(x, what, context).toIntUnchecked();
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
}
