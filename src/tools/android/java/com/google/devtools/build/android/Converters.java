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
import com.android.builder.core.VariantTypeImpl;
import com.android.repository.Revision;
import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.converters.IParameterSplitter;
import com.beust.jcommander.converters.StringConverter;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import java.io.File;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Some convenient converters used by android actions. Note: These are specific to android actions.
 */
public final class Converters {
  /**
   * Converter for {@link UnvalidatedAndroidData}. Relies on {@code
   * UnvalidatedAndroidData#valueOf(String)} to perform conversion and validation. Compatible with
   * JCommander.
   */
  public static class CompatUnvalidatedAndroidDataConverter
      implements IStringConverter<UnvalidatedAndroidData> {
    @Override
    public UnvalidatedAndroidData convert(String input) throws ParameterException {
      try {
        return UnvalidatedAndroidData.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new ParameterException(
            String.format("invalid UnvalidatedAndroidData: %s", e.getMessage()), e);
      }
    }
  }

  /** Converter for {@link UnvalidatedAndroidDirectories}. Compatible with JCommander. */
  public static class CompatUnvalidatedAndroidDirectoriesConverter
      implements IStringConverter<UnvalidatedAndroidDirectories> {
    @Override
    public UnvalidatedAndroidDirectories convert(String input) throws ParameterException {
      try {
        return UnvalidatedAndroidDirectories.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new ParameterException(
            String.format("invalid UnvalidatedAndroidDirectories: %s", e.getMessage()), e);
      }
    }
  }

  /** Converter for {@link DependencyAndroidData}. Compatible with JCommander. */
  public static class CompatDependencyAndroidDataConverter
      implements IStringConverter<DependencyAndroidData> {
    @Override
    public DependencyAndroidData convert(String input) throws ParameterException {
      try {
        return DependencyAndroidData.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new ParameterException(
            String.format("invalid DependencyAndroidData: %s", e.getMessage()), e);
      }
    }
  }

  /** Converter for a single {@link SerializedAndroidData}. Compatible with JCommander. */
  public static class CompatSerializedAndroidDataConverter
      implements IStringConverter<SerializedAndroidData> {
    @Override
    public SerializedAndroidData convert(String input) throws ParameterException {
      try {
        return SerializedAndroidData.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new ParameterException(
            String.format("invalid SerializedAndroidData: %s", e.getMessage()), e);
      }
    }
  }

  /** A splitter class for JCommander flags that splits on ampersands ("&"). */
  public static class AmpersandSplitter implements IParameterSplitter {
    @Override
    public List<String> split(String value) {
      if (value.isEmpty()) {
        return ImmutableList.of();
      }
      return ImmutableList.copyOf(value.split("&"));
    }
  }

  /** A splitter class for JCommander flags that splits on colons (":"). */
  public static class ColonSplitter implements IParameterSplitter {
    @Override
    public List<String> split(String value) {
      if (value.isEmpty()) {
        return ImmutableList.of();
      }
      return ImmutableList.copyOf(value.split(":"));
    }
  }

  /**
   * A splitter class for JCommander flags that does not actually split.
   *
   * <p>Used when an argument expects a comma in the value (such as DependencySymbolFileProvider).
   */
  public static class NoOpSplitter implements IParameterSplitter {
    @Override
    public List<String> split(String value) {
      if (value.isEmpty()) {
        return ImmutableList.of();
      }
      return ImmutableList.of(value);
    }
  }

  /** Converter for a single {@link DependencySymbolFileProvider}. Compatible with JCommander. */
  public static class CompatDependencySymbolFileProviderConverter
      implements IStringConverter<DependencySymbolFileProvider> {

    @Override
    public DependencySymbolFileProvider convert(String input) throws ParameterException {
      try {
        return DependencySymbolFileProvider.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new ParameterException(
            String.format("invalid DependencyAndroidData: %s", e.getMessage()), e);
      }
    }
  }

  /**
   * Converter for {@link Revision}. Relies on {@code Revision#parseRevision(String)} to perform
   * conversion and validation. Compatible with JCommander.
   */
  public static class CompatRevisionConverter implements IStringConverter<Revision> {

    @Override
    public Revision convert(String input) throws ParameterException {
      try {
        return Revision.parseRevision(input);
      } catch (NumberFormatException e) {
        throw new ParameterException(e.getMessage());
      }
    }
  }

  /** Converter class for _existing_ `Path`-s that is compatible with JCommander. */
  public static class CompatExistingPathConverter extends CompatPathConverter {
    public CompatExistingPathConverter() {
      super(true);
    }
  }

  /** Converter class for `Path`-s that is compatible with JCommander. */
  public static class CompatPathConverter implements IStringConverter<Path> {
    private final boolean mustExist;

    public CompatPathConverter() {
      this.mustExist = false;
    }

    protected CompatPathConverter(boolean mustExist) {
      this.mustExist = mustExist;
    }

    @Override
    public Path convert(String input) throws ParameterException {
      // Below snippet is cribbed from PathConverter.convert(). The only difference is that this
      // throws a ParameterException instead of an OptionsParsingException.
      try {
        Path path = FileSystems.getDefault().getPath(input);
        if (mustExist && !Files.exists(path)) {
          throw new ParameterException(
              String.format("%s is not a valid path: it does not exist.", input));
        }
        return path;
      } catch (InvalidPathException e) {
        throw new ParameterException(
            String.format("%s is not a valid path: %s.", input, e.getMessage()), e);
      }
    }
  }

  /** Converter for {@link VariantType}. Compatible with JCommander. */
  public static class CompatVariantTypeConverter implements IStringConverter<VariantTypeImpl> {
    @Override
    public VariantTypeImpl convert(String input) throws ParameterException {
      try {
        return VariantTypeImpl.valueOf(input);
      } catch (IllegalArgumentException e) {
        throw new ParameterException(String.format("invalid VariantType: %s", e.getMessage()), e);
      }
    }
  }

  /**
   * Validating converter for a list of Paths. A Path is considered valid if it resolves to a file.
   *
   * <p>Compatible with JCommander.
   */
  @Deprecated // Not _actually_ deprecated (see cl/162194755).
  public static class CompatPathListConverter implements IStringConverter<List<Path>> {
    private final CompatPathConverter baseConverter;

    public CompatPathListConverter() {
      this(false);
    }

    public CompatPathListConverter(boolean mustExist) {
      this.baseConverter = new CompatPathConverter(mustExist);
    }

    @Override
    public List<Path> convert(String input) throws ParameterException {
      List<Path> list = new ArrayList<>();
      for (String piece : input.split(":")) {
        if (!piece.isEmpty()) {
          list.add(baseConverter.convert(piece));
        }
      }
      return Collections.unmodifiableList(list);
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
   * Converts args of format key;value[,key;value]*. Most of the logic is recycled from
   * DictionaryConverter. The main difference here is compatibility with JCommander's converter
   * interface (IStringConverter) and its primary exception class (ParameterException).
   */
  public abstract static class CompatDictionaryConverter<K, V>
      implements IStringConverter<Map<K, V>> {
    IStringConverter<K> keyConverter;
    IStringConverter<V> valueConverter;

    public CompatDictionaryConverter(
        IStringConverter<K> keyConverter, IStringConverter<V> valueConverter) {
      this.keyConverter = keyConverter;
      this.valueConverter = valueConverter;
    }

    @Override
    public Map<K, V> convert(String input) throws ParameterException {
      // This method is cribbed from {@code DictionaryConverter.convert()}. The only differences are
      // that this throws a ParameterException instead of an OptionsParsingException, and
      // JCommander's {@code IStringConverter<>} is used instead of {@code Converter<>}.
      if (input.isEmpty()) {
        return ImmutableMap.of();
      }
      Map<K, V> map = new LinkedHashMap<>();
      // Only split on comma and colon that are not escaped with a backslash
      for (String entry : input.split(UNESCAPED_COMMA_REGEX)) {
        String[] entryFields = entry.split(UNESCAPED_COLON_REGEX, -1);
        if (entryFields.length < 2) {
          throw new ParameterException(
              String.format(
                  "Dictionary entry [%s] does not contain both a key and a value.", entry));
        } else if (entryFields.length > 2) {
          throw new ParameterException(
              String.format("Dictionary entry [%s] contains too many fields.", entry));
        }
        // Unescape any comma or colon that is not a key or value separator.
        String keyString = unescapeInput(entryFields[0]);
        K key = keyConverter.convert(keyString);
        if (map.containsKey(keyString)) {
          throw new ParameterException(
              String.format("Dictionary already contains the key [%s].", keyString));
        }
        // Unescape any comma or colon that is not a key or value separator.
        String valueString = unescapeInput(entryFields[1]);
        V value = valueConverter.convert(valueString);
        map.put(key, value);
      }
      return ImmutableMap.copyOf(map);
    }
  }

  /**
   * Converts dictionary args for {@code Map<String, String>}, compatible with JCommander. Should be
   * backward compatible with StringDictionaryConverter.
   */
  public static class CompatStringDictionaryConverter
      extends CompatDictionaryConverter<String, String> {
    public CompatStringDictionaryConverter() {
      super(new StringConverter(), new StringConverter());
    }
  }

  /**
   * Converts dictionary args for {@code Map<Path, String>}, compatible with JCommander. Should be
   * backward compatible with ExistingPathStringDictionaryConverter.
   */
  public static class CompatExistingPathStringDictionaryConverter
      extends CompatDictionaryConverter<Path, String> {
    public CompatExistingPathStringDictionaryConverter() {
      super(new CompatExistingPathConverter(), new StringConverter());
    }
  }

  /** Converts a list of static library strings into paths. Compatible with JCommander. */
  @Deprecated
  public static class CompatStaticLibraryListConverter
      implements IStringConverter<List<StaticLibrary>> {
    static final Splitter SPLITTER = Splitter.on(File.pathSeparatorChar);

    static final CompatStaticLibraryConverter libraryConverter = new CompatStaticLibraryConverter();

    @Override
    public List<StaticLibrary> convert(String input) throws ParameterException {
      final ImmutableList.Builder<StaticLibrary> builder = ImmutableList.<StaticLibrary>builder();
      for (String path : SPLITTER.splitToList(input)) {
        builder.add(libraryConverter.convert(path));
      }
      return builder.build();
    }
  }

  /** Converts a list of static library strings into paths. Compatible with JCommander. */
  public static class CompatStaticLibraryConverter implements IStringConverter<StaticLibrary> {
    static final CompatPathConverter pathConverter = new CompatPathConverter(true);

    @Override
    public StaticLibrary convert(String input) throws ParameterException {
      return StaticLibrary.from(pathConverter.convert(input));
    }
  }

  /** Converts a string of resources and manifest into paths. Compatible with JCommander. */
  public static class CompatCompiledResourcesConverter
      implements IStringConverter<CompiledResources> {
    static final CompatPathConverter pathConverter = new CompatPathConverter(true);
    static final Pattern COMPILED_RESOURCE_FORMAT = Pattern.compile("(.+):(.+)");

    @Override
    public CompiledResources convert(String input) throws ParameterException {
      final Matcher matched = COMPILED_RESOURCE_FORMAT.matcher(input);
      if (!matched.find()) {
        throw new ParameterException("Expected format <resources zip>:<manifest>");
      }
      Path resources = pathConverter.convert(matched.group(1));
      Path manifest = pathConverter.convert(matched.group(2));
      return CompiledResources.from(resources, manifest);
    }
  }
}
