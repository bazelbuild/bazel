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

package com.google.devtools.build.lib.util;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.StandardSystemProperty;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.ParsedOptionDescription;
import java.util.List;
import javax.annotation.Nullable;

/** Blaze-specific option utilities. */
public final class OptionsUtils {

  /**
   * Returns a string representation of the non-hidden specified options; option values are
   * shell-escaped.
   */
  public static String asShellEscapedString(Iterable<ParsedOptionDescription> optionsList) {
    StringBuilder result = new StringBuilder();
    for (ParsedOptionDescription option : optionsList) {
      if (option.isHidden()) {
        continue;
      }
      if (result.length() != 0) {
        result.append(' ');
      }
      result.append(option.getCanonicalFormWithValueEscaper(ShellEscaper::escapeString));
    }
    return result.toString();
  }

  /**
   * Returns a string representation of the non-hidden explicitly or implicitly specified options;
   * option values are shell-escaped.
   */
  public static String asShellEscapedString(OptionsParsingResult options) {
    return asShellEscapedString(options.asCompleteListOfParsedOptions());
  }

  /**
   * Return a representation of the non-hidden specified options, as a list of string. No escaping
   * is done.
   */
  public static List<String> asArgumentList(Iterable<ParsedOptionDescription> optionsList) {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (ParsedOptionDescription option : optionsList) {
      if (option.isHidden()) {
        continue;
      }
      builder.add(option.getCanonicalForm());
    }
    return builder.build();
  }

  /**
   * Return a representation of the non-hidden specified options, as a list of string. No escaping
   * is done.
   */
  public static List<String> asArgumentList(OptionsParsingResult options) {
    return asArgumentList(options.asCompleteListOfParsedOptions());
  }

  /**
   * Returns a string representation of the non-hidden explicitly or implicitly specified options,
   * filtering out any sensitive options; option values are shell-escaped.
   */
  public static String asFilteredShellEscapedString(
      OptionsParsingResult options, Iterable<ParsedOptionDescription> optionsList) {
    return asShellEscapedString(optionsList);
  }

  /**
   * Returns a string representation of the non-hidden explicitly or implicitly specified options,
   * filtering out any sensitive options; option values are shell-escaped.
   */
  public static String asFilteredShellEscapedString(OptionsParsingResult options) {
    return asFilteredShellEscapedString(options, options.asCompleteListOfParsedOptions());
  }

  /** Converter from String to PathFragment. */
  public static class PathFragmentConverter extends Converter.Contextless<PathFragment> {

    @Override
    public PathFragment convert(String input) {
      return convertOptionsPathFragment(checkNotNull(input));
    }

    @Override
    public String getTypeDescription() {
      return "a path";
    }
  }

  /** Converter from String to PathFragment. If the input is empty returns {@code null} instead. */
  public static class EmptyToNullPathFragmentConverter extends Converter.Contextless<PathFragment> {

    @Override
    @Nullable
    public PathFragment convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        return null;
      }
      return convertOptionsPathFragment(input);
    }

    @Override
    public String getTypeDescription() {
      return "a path";
    }
  }

  /** Converter from String to PathFragment requiring the provided path to be absolute. */
  public static class AbsolutePathFragmentConverter extends Converter.Contextless<PathFragment> {

    @Override
    public PathFragment convert(String input) throws OptionsParsingException {
      PathFragment parsed = convertOptionsPathFragment(checkNotNull(input));
      if (!parsed.isAbsolute()) {
        throw new OptionsParsingException(String.format("Not an absolute path: '%s'", input));
      }
      return parsed;
    }

    @Override
    public String getTypeDescription() {
      return "an absolute path";
    }
  }

  /** Converts from a colon-separated list of strings into a list of PathFragment instances. */
  public static class PathFragmentListConverter
      extends Converter.Contextless<ImmutableList<PathFragment>> {

    @Override
    public ImmutableList<PathFragment> convert(String input) {
      ImmutableList.Builder<PathFragment> result = ImmutableList.builder();
      for (String piece : input.split(":")) {
        if (!piece.isEmpty()) {
          result.add(convertOptionsPathFragment(piece));
        }
      }
      return result.build();
    }

    @Override
    public String getTypeDescription() {
      return "a colon-separated list of paths";
    }
  }

  private static PathFragment convertOptionsPathFragment(String path) {
    if (!path.isEmpty() && path.startsWith("~/")) {
      path = path.replace("~", StandardSystemProperty.USER_HOME.value());
    }
    return PathFragment.create(path);
  }
}
