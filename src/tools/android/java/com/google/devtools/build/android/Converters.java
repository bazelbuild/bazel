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
package com.google.devtools.build.android;

import com.android.builder.core.VariantType;
import com.android.manifmerger.ManifestMerger2;
import com.android.manifmerger.ManifestMerger2.MergeType;
import com.android.repository.Revision;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.File;
import java.lang.reflect.ParameterizedType;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Some convenient converters used by android actions. Note: These are specific to android actions.
 */
public final class Converters {
  private static final Converter<String> IDENTITY_CONVERTER = new Converter<String>() {
    @Override public String convert(String input) {
      return input;
    }

    @Override public String getTypeDescription() {
      return "a string";
    }
  };

  /**
   * Converter for {@link UnvalidatedAndroidData}. Relies on
   * {@code UnvalidatedAndroidData#valueOf(String)} to perform conversion and validation.
   */
  public static class UnvalidatedAndroidDataConverter implements Converter<UnvalidatedAndroidData> {

    @Override
    public UnvalidatedAndroidData convert(String input) throws OptionsParsingException {
      try {
        return UnvalidatedAndroidData.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("invalid UnvalidatedAndroidData: %s", e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "unvalidated android data in the format " + UnvalidatedAndroidData.EXPECTED_FORMAT;
    }
  }

  /**
   * Converter for {@link UnvalidatedAndroidDirectories}.
   */
  public static class UnvalidatedAndroidDirectoriesConverter
      implements Converter<UnvalidatedAndroidDirectories> {

    @Override
    public UnvalidatedAndroidDirectories convert(String input) throws OptionsParsingException {
      try {
        return UnvalidatedAndroidDirectories.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("invalid UnvalidatedAndroidDirectories: %s", e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "unvalidated android directories in the format "
          + UnvalidatedAndroidDirectories.EXPECTED_FORMAT;
    }
  }

  /**
   * Converter for a list of {@link DependencyAndroidData}. Relies on
   * {@code DependencyAndroidData#valueOf(String)} to perform conversion and validation.
   */
  public static class DependencyAndroidDataListConverter
      implements Converter<List<DependencyAndroidData>> {

    @Override
    public List<DependencyAndroidData> convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        return ImmutableList.of();
      }
      try {
        ImmutableList.Builder<DependencyAndroidData> builder = ImmutableList.builder();
        for (String item : input.split(",")) {
          builder.add(DependencyAndroidData.valueOf(item));
        }
        return builder.build();
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("invalid DependencyAndroidData: %s", e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a list of dependency android data in the format "
          + DependencyAndroidData.EXPECTED_FORMAT + "[,...]";
    }
  }

  /**
   * Converter for a {@link SerializedAndroidData}.
   */
  public static class SerializedAndroidDataConverter implements Converter<SerializedAndroidData> {

    @Override
    public SerializedAndroidData convert(String input) throws OptionsParsingException {
      try {
        return SerializedAndroidData.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("invalid SerializedAndroidData: %s", e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "preparsed android data in the format " + SerializedAndroidData.EXPECTED_FORMAT;
    }
  }

  /**
   * Converter for a list of {@link SerializedAndroidData}.
   */
  public static class SerializedAndroidDataListConverter
      implements Converter<List<SerializedAndroidData>> {

    @Override
    public List<SerializedAndroidData> convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        return ImmutableList.of();
      }
      try {
        ImmutableList.Builder<SerializedAndroidData> builder = ImmutableList.builder();
        for (String entry : input.split("&")) {
          builder.add(SerializedAndroidData.valueOf(entry));
        }
        return builder.build();
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("invalid SerializedAndroidData: %s", e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a list of preparsed android data in the format "
          + SerializedAndroidData.EXPECTED_FORMAT + "[&...]";
    }
  }

  /**
   * Converter for a list of {@link DependencySymbolFileProvider}. Relies on
   * {@code DependencySymbolFileProvider#valueOf(String)} to perform conversion and validation.
   */
  public static class DependencySymbolFileProviderListConverter
      implements Converter<List<DependencySymbolFileProvider>> {

    @Override
    public List<DependencySymbolFileProvider> convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        return ImmutableList.<DependencySymbolFileProvider>of();
      }
      try {
        ImmutableList.Builder<DependencySymbolFileProvider> builder = ImmutableList.builder();
        for (String item : input.split(",")) {
          builder.add(DependencySymbolFileProvider.valueOf(item));
        }
        return builder.build();
      } catch (IllegalArgumentException e) {
        throw new OptionsParsingException(
            String.format("invalid DependencyAndroidData: %s", e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return String.format("a list of dependency android data in the format: %s[%s]",
          DependencySymbolFileProvider.commandlineFormat("1"),
          DependencySymbolFileProvider.commandlineFormat("2"));
    }
  }

  /**
   * Converter for {@link Revision}. Relies on {@code Revision#parseRevision(String)} to
   * perform conversion and validation.
   */
  public static class RevisionConverter implements Converter<Revision> {

    @Override
    public Revision convert(String input) throws OptionsParsingException {
      try {
        return Revision.parseRevision(input);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException(e.getMessage());
      }
    }

    @Override
    public String getTypeDescription() {
      return "a revision number";
    }
  }

  /** Validating converter for Paths. A Path is considered valid if it resolves to a file. */
  public static class PathConverter implements Converter<Path> {

    private final boolean mustExist;

    public PathConverter() {
      this.mustExist = false;
    }

    protected PathConverter(boolean mustExist) {
      this.mustExist = mustExist;
    }

    @Override
    public Path convert(String input) throws OptionsParsingException {
      try {
        Path path = FileSystems.getDefault().getPath(input);
        if (mustExist && !Files.exists(path)) {
          throw new OptionsParsingException(
              String.format("%s is not a valid path: it does not exist.", input));
        }
        return path;
      } catch (InvalidPathException e) {
        throw new OptionsParsingException(
            String.format("%s is not a valid path: %s.", input, e.getMessage()), e);
      }
    }

    @Override
    public String getTypeDescription() {
      return "a valid filesystem path";
    }
  }

  /**
   * Validating converter for Paths. A Path is considered valid if it resolves to a file and exists.
   */
  public static class ExistingPathConverter extends PathConverter {
    public ExistingPathConverter() {
      super(true);
    }
  }

  /** Converter for {@link VariantType}. */
  public static class VariantTypeConverter extends EnumConverter<VariantType> {
    public VariantTypeConverter() {
      super(VariantType.class, "variant type");
    }
  }

  /** Converter for {@link ManifestMerger2}.{@link MergeType}. */
  public static class MergeTypeConverter
      extends EnumConverter<MergeType> {
    public MergeTypeConverter() {
      super(MergeType.class, "merge type");
    }
  }

  /**
   * Validating converter for a list of Paths.
   * A Path is considered valid if it resolves to a file.
   */
  @Deprecated
  public static class PathListConverter implements Converter<List<Path>> {

    public static List<Path> concatLists(@Nullable List<Path> a, @Nullable List<Path> b) {
      if (a == null || a.isEmpty()) {
        return (b == null || b.isEmpty()) ? ImmutableList.of() : b;
      }
      return (b == null || b.isEmpty()) ? a : ImmutableList.copyOf(Iterables.concat(a, b));
    }

    private final PathConverter baseConverter;

    public PathListConverter() {
      this(false);
    }

    protected PathListConverter(boolean mustExist) {
      baseConverter = new PathConverter(mustExist);
    }

    @Override
    public List<Path> convert(String input) throws OptionsParsingException {
      List<Path> list = new ArrayList<>();
      for (String piece : input.split(File.pathSeparator)) {
        if (!piece.isEmpty()) {
          list.add(baseConverter.convert(piece));
        }
      }
      return Collections.unmodifiableList(list);
    }

    @Override
    public String getTypeDescription() {
      return "a colon-separated list of paths";
    }
  }

  // Commas that are not escaped by a backslash.
  private static final String UNESCAPED_COMMA_REGEX = "(?<!\\\\)\\,";
  // Colons that are not escaped by a backslash.
  private static final String UNESCAPED_COLON_REGEX = "(?<!\\\\)\\:";

  private static String unescapeInput(String input) {
    return input.replace("\\:", ":").replace("\\,", ",");
  }

  /**
   * A converter for dictionary arguments of the format key:value[,key:value]*. The keys and values
   * may contain colons and commas as long as they are escaped with a backslash.
   */
  private abstract static class DictionaryConverter<K, V> implements Converter<Map<K, V>> {
    private final Converter<K> keyConverter;
    private final Converter<V> valueConverter;

    public DictionaryConverter(Converter<K> keyConverter, Converter<V> valueConverter) {
      this.keyConverter = keyConverter;
      this.valueConverter = valueConverter;
    }

    @Override
    public Map<K, V> convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        return ImmutableMap.of();
      }
      Map<K, V> map = new LinkedHashMap<>();
      // Only split on comma and colon that are not escaped with a backslash
      for (String entry : input.split(UNESCAPED_COMMA_REGEX)) {
        String[] entryFields = entry.split(UNESCAPED_COLON_REGEX, -1);
        if (entryFields.length < 2) {
          throw new OptionsParsingException(String.format(
              "Dictionary entry [%s] does not contain both a key and a value.",
              entry));
        } else if (entryFields.length > 2) {
          throw new OptionsParsingException(String.format(
              "Dictionary entry [%s] contains too many fields.",
              entry));
        }
        // Unescape any comma or colon that is not a key or value separator.
        String keyString = unescapeInput(entryFields[0]);
        K key = keyConverter.convert(keyString);
        if (map.containsKey(key)) {
          throw new OptionsParsingException(String.format(
              "Dictionary already contains the key [%s].",
              keyString));
        }
        // Unescape any comma or colon that is not a key or value separator.
        String valueString = unescapeInput(entryFields[1]);
        V value = valueConverter.convert(valueString);
        map.put(key, value);
      }
      return ImmutableMap.copyOf(map);
    }

    @Override
    public String getTypeDescription() {
      // Retrieve types of dictionary through reflection to avoid overriding this method in each
      // subclass or passing types to this superclass.
      return String.format(
          "a comma-separated list of colon-separated key value pairs of the types %s and %s",
          ((ParameterizedType) getClass().getGenericSuperclass()).getActualTypeArguments()[0],
          ((ParameterizedType) getClass().getGenericSuperclass()).getActualTypeArguments()[1]);
    }
  }

  /**
   * A converter for dictionary arguments of the format key:value[,key:value]*. The keys and values
   * may contain colons and commas as long as they are escaped with a backslash. The key and value
   * types are both String.
   */
  public static class StringDictionaryConverter extends DictionaryConverter<String, String> {
    public StringDictionaryConverter() {
      super(IDENTITY_CONVERTER, IDENTITY_CONVERTER);
    }
    // The way {@link OptionsData} checks for generic types requires convert to have literal type
    // parameters and not argument type parameters.
    @Override public Map<String, String> convert(String input) throws OptionsParsingException {
      return super.convert(input);
    }
  }

  /**
   * A converter for dictionary arguments of the format key:value[,key:value]*. The keys and values
   * may contain colons and commas as long as they are escaped with a backslash. The key type is
   * Path and the value type is String.
   */
  public static class ExistingPathStringDictionaryConverter
      extends DictionaryConverter<Path, String> {
    public ExistingPathStringDictionaryConverter() {
      super(new ExistingPathConverter(), IDENTITY_CONVERTER);
    }
    // The way {@link OptionsData} checks for generic types requires convert to have literal type
    // parameters and not argument type parameters.
    @Override public Map<Path, String> convert(String input) throws OptionsParsingException {
      return super.convert(input);
    }
  }
}
