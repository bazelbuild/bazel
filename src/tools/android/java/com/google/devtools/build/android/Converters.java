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

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsParsingException;

import com.android.builder.core.VariantConfiguration;
import com.android.builder.core.VariantConfiguration.Type;
import com.android.sdklib.repository.FullRevision;

import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Some convenient converters used by android actions. Note: These are specific to android actions.
 */
public final class Converters {
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
      return "unvalidated android data in the format "
          + "resources[#resources]:assets[#assets]:manifest";
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
        return ImmutableList.<DependencyAndroidData>of();
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
          + "resources[#resources]:assets[#assets]:manifest:r.txt"
          + "[,resources[#resources]:assets[#assets]:manifest:r.txt]";
    }
  }

  /**
   * Converter for {@link FullRevision}. Relies on {@code FullRevision#parseRevision(String)} to
   * perform conversion and validation.
   */
  public static class FullRevisionConverter implements Converter<FullRevision> {

    @Override
    public FullRevision convert(String input) throws OptionsParsingException {
      try {
        return FullRevision.parseRevision(input);
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

  /** Converter for {@link VariantConfiguration}.{@link Type}. */
  public static class VariantConfigurationTypeConverter
      extends EnumConverter<VariantConfiguration.Type> {
    public VariantConfigurationTypeConverter() {
      super(VariantConfiguration.Type.class, "variant configuration type");
    }
  }

  /**
   * Validating converter for a list of Paths.
   * A Path is considered valid if it resolves to a file.
   */
  public static class PathListConverter implements Converter<List<Path>> {

    final PathConverter baseConverter = new PathConverter();

    @Override
    public List<Path> convert(String input) throws OptionsParsingException {
      List<Path> list = new ArrayList<>();
      for (String piece : input.split(":")) {
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

}
