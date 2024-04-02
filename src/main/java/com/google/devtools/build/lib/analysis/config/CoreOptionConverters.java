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

package com.google.devtools.build.lib.analysis.config;

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.BooleanConverter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Converters.StringConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkInt;

/**
 * {@link Converter}s for {@link com.google.devtools.common.options.Option}s that aren't
 * domain-specific (i.e. aren't consumed within a single {@link FragmentOptions}).
 */
public class CoreOptionConverters {

  // Not instantiable.
  private CoreOptionConverters() {}

  /**
   * The set of converters used for {@link com.google.devtools.build.lib.packages.BuildSetting}
   * value parsing.
   */
  public static final ImmutableMap<Type<?>, Converter<?>> BUILD_SETTING_CONVERTERS =
      new ImmutableMap.Builder<Type<?>, Converter<?>>()
          .put(INTEGER, new StarlarkIntConverter())
          .put(BOOLEAN, new BooleanConverter())
          .put(STRING, new StringConverter())
          .put(STRING_LIST, new CommaSeparatedOptionListConverter())
          .put(LABEL, new LabelConverter())
          .put(LABEL_LIST, new LabelListConverter())
          .put(NODEP_LABEL, new LabelConverter())
          .buildOrThrow();

  /** A converter from strings to Starlark int values. */
  private static class StarlarkIntConverter extends Converter.Contextless<StarlarkInt> {
    @Override
    public StarlarkInt convert(String input) throws OptionsParsingException {
      // Note that Starlark rule attribute values are currently restricted
      // to the signed 32-bit range, but Starlark-based flags may take on
      // any integer value.
      try {
        return StarlarkInt.parse(input, 0);
      } catch (NumberFormatException ex) {
        throw new OptionsParsingException("invalid int: " + ex.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "an int";
    }
  }

  /** A converter from strings to Labels. */
  public static class LabelConverter implements Converter<Label> {
    @Override
    public Label convert(String input, Object conversionContext) throws OptionsParsingException {
      return convertOptionsLabel(input, conversionContext);
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /** A converter from comma-separated strings to Label lists. */
  public static class LabelListConverter implements Converter<List<Label>> {
    @Override
    public List<Label> convert(String input, Object conversionContext)
        throws OptionsParsingException {
      ImmutableList.Builder<Label> result = ImmutableList.builder();
      for (String label : Splitter.on(",").omitEmptyStrings().split(input)) {
        result.add(convertOptionsLabel(label, conversionContext));
      }
      return result.build();
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /**
   * A converter from comma-separated strings to Labels which preserves order, but in case of
   * duplicates, keeps only the first copy.
   */
  public static class LabelOrderedSetConverter extends LabelListConverter {
    @Override
    public List<Label> convert(String input, Object conversionContext)
        throws OptionsParsingException {
      Set<Label> alreadySeen = new HashSet<>();
      ImmutableList.Builder<Label> result = ImmutableList.builder();
      for (Label label : super.convert(input, conversionContext)) {
        if (alreadySeen.add(label)) {
          result.add(label);
        }
      }
      return result.build();
    }
  }

  /**
   * A converter that returns null if the input string is empty, otherwise it converts the input to
   * a label.
   */
  public static class EmptyToNullLabelConverter implements Converter<Label> {
    @Override
    @Nullable
    public Label convert(String input, Object conversionContext) throws OptionsParsingException {
      return input.isEmpty() ? null : convertOptionsLabel(input, conversionContext);
    }

    @Override
    public String getTypeDescription() {
      return "a build target label";
    }
  }

  /** Flag converter for a map of unique keys with optional labels as values. */
  public static class LabelMapConverter implements Converter<Map<String, Label>> {
    @Override
    public Map<String, Label> convert(String input, Object conversionContext)
        throws OptionsParsingException {
      // Use LinkedHashMap so we can report duplicate keys more easily while preserving order
      Map<String, Label> result = new LinkedHashMap<>();
      for (String entry : Splitter.on(",").omitEmptyStrings().trimResults().split(input)) {
        String key;
        Label label;
        int sepIndex = entry.indexOf('=');
        if (sepIndex < 0) {
          key = entry;
          label = null;
        } else {
          key = entry.substring(0, sepIndex);
          String value = entry.substring(sepIndex + 1);
          label = value.isEmpty() ? null : convertOptionsLabel(value, conversionContext);
        }
        if (result.containsKey(key)) {
          throw new OptionsParsingException("Key '" + key + "' appears twice");
        }
        result.put(key, label);
      }
      return Collections.unmodifiableMap(result);
    }

    @Override
    public boolean starlarkConvertible() {
      return true;
    }

    @Override
    public String reverseForStarlark(Object converted) {
      @SuppressWarnings("unchecked")
      Map<String, Label> typedValue = (Map<String, Label>) converted;
      return typedValue.entrySet().stream()
          .map(
              e ->
                  e.getValue() == null
                      ? e.getKey()
                      : String.format("%s=%s", e.getKey(), e.getValue()))
          .collect(joining(","));
    }

    @Override
    public String getTypeDescription() {
      return "a comma-separated list of keys optionally followed by '=' and a label";
    }
  }

  /** Flag converter for assigning a Label to a String. */
  public static class LabelToStringEntryConverter implements Converter<Map.Entry<Label, String>> {
    @Override
    public Map.Entry<Label, String> convert(String input, Object conversionContext)
        throws OptionsParsingException {
      // TODO(twigg): This doesn't work well if the labels can themselves have an '='
      long equalsCount = input.chars().filter(c -> c == '=').count();
      if (equalsCount != 1 || input.charAt(0) == '=' || input.charAt(input.length() - 1) == '=') {
        throw new OptionsParsingException(
            "Variable definitions must be in the form of a 'name=value' assignment. 'name' and"
                + " 'value' must be non-empty and may not include '='.");
      }
      int pos = input.indexOf("=");
      Label name = convertOptionsLabel(input.substring(0, pos), conversionContext);
      String value = input.substring(pos + 1);
      return Maps.immutableEntry(name, value);
    }

    @Override
    public String getTypeDescription() {
      return "a 'label=value' assignment";
    }
  }

  /** Values for the --strict_*_deps option */
  public enum StrictDepsMode {
    /** Silently allow referencing transitive dependencies. */
    OFF,
    /** Warn about transitive dependencies being used directly. */
    WARN,
    /** Fail the build when transitive dependencies are used directly. */
    ERROR,
    /** Transition to strict by default. */
    STRICT,
    /** When no flag value is specified on the command line. */
    DEFAULT
  }

  /** Converter for the --strict_*_deps option. */
  public static class StrictDepsConverter extends EnumConverter<StrictDepsMode> {
    public StrictDepsConverter() {
      super(StrictDepsMode.class, "strict dependency checking level");
    }
  }

  private static Label convertOptionsLabel(String input, @Nullable Object conversionContext)
      throws OptionsParsingException {
    try {
      if (conversionContext instanceof Label.PackageContext) {
        // This can happen if this converter is being used to convert flag values specified in
        // Starlark, for example in a transition implementation function.
        return Label.parseWithPackageContext(input, (Label.PackageContext) conversionContext);
      }
      // Check if the input starts with '/'. We don't check for "//" so that
      // we get a better error message if the user accidentally tries to use
      // an absolute path (starting with '/') for a label.
      if (!input.startsWith("/") && !input.startsWith("@")) {
        input = "//" + input;
      }
      if (conversionContext == null) {
        // This can happen in the first round of option parsing, before repo mappings are
        // calculated. In this case, it actually doesn't matter how we parse label-typed flags, as
        // they shouldn't be used anywhere anyway.
        return Label.parseCanonical(input);
      }
      Preconditions.checkArgument(
          conversionContext instanceof RepositoryMapping,
          "bad conversion context type: %s",
          conversionContext.getClass().getName());
      // This can happen in the second round of option parsing.
      return Label.parseWithRepoContext(
          input, Label.RepoContext.of(RepositoryName.MAIN, (RepositoryMapping) conversionContext));
    } catch (LabelSyntaxException e) {
      throw new OptionsParsingException(e.getMessage());
    }
  }
}
