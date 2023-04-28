// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.IValueValidator;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Stopwatch;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.singlejar.ZipCombiner.OutputMode;
import com.google.devtools.build.zip.ZipFileEntry;
import com.google.devtools.build.zip.ZipReader;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Action to filter entries out of a Zip file.
 *
 * <p>The entries to remove are determined from the filterZips and filterTypes. All entries from the
 * filter Zip files that have an extension listed in filterTypes will be removed. If no filterZips
 * are specified, no entries will be removed. Specifying no filterTypes is treated as if an
 * extension of '.*' was specified.
 *
 * <p>Assuming each Zip as a set of entries, the result is:
 *
 * <pre> outputZip = inputZip - union[x intersect filterTypes for x in filterZips]</pre>
 *
 * <p>
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/ZipFilterAction\
 *      --inputZip path/to/inputZip
 *      --outputZip path/to/outputZip
 *      --filterZips [path/to/filterZip[,path/to/filterZip]...]
 *      --filterTypes [fileExtension[,fileExtension]...]
 *      --explicitFilters [fileRegex[,fileRegex]...]
 *      --outputMode [DONT_CARE|FORCE_DEFLATE|FORCE_STORED]
 *      --checkHashMismatch [IGNORE|WARN|ERROR]
 * </pre>
 */
public class ZipFilterAction {

  private static final Logger logger = Logger.getLogger(ZipFilterAction.class.getName());

  /** Modes of performing content hash checking during zip filtering. */
  public enum HashMismatchCheckMode {
    /** Filter file from input zip iff a file is found with the same filename in filter zips. */
    IGNORE,

    /**
     * Filter file from input zip iff a file is found with the same filename and content hash in
     * filter zips. Print warning if the filename is identical but content hash is not.
     */
    WARN,

    /**
     * Same behavior as WARN, but throw an error if a file is found with the same filename with
     * different content hash.
     */
    ERROR
  }

  @Parameters()
  static class Options {
    @Parameter(
      names = "--inputZip",
      description = "Path of input zip.",
      converter = PathFlagConverter.class,
      validateValueWith = PathExistsValidator.class
    )
    Path inputZip;

    @Parameter(
      names = "--outputZip",
      description = "Path to write output zip.",
      converter = PathFlagConverter.class
    )
    Path outputZip;

    @Parameter(
      names = "--filterZips",
      description = "Filter zips.",
      converter = PathFlagConverter.class,
      validateValueWith = AllPathsExistValidator.class
    )
    List<Path> filterZips = ImmutableList.of();

    @Parameter(names = "--filterTypes", description = "Filter file types.")
    List<String> filterTypes = ImmutableList.of();

    @Parameter(names = "--explicitFilters", description = "Explicitly specified filters.")
    List<String> explicitFilters = ImmutableList.of();

    @Parameter(names = "--outputMode", description = "Output zip compression mode.")
    OutputMode outputMode = OutputMode.DONT_CARE;

    @Parameter(
      names = "--checkHashMismatch",
      description =
          "Ignore, warn or throw an error if the content hashes of two files with the "
              + "same name are different."
    )
    HashMismatchCheckMode hashMismatchCheckMode = HashMismatchCheckMode.WARN;

    /**
     * @deprecated please use --checkHashMismatch ERROR instead. Other options are IGNORE and WARN.
     */
    @Deprecated
    @Parameter(
      names = "--errorOnHashMismatch",
      description = "Error on entry filter with hash mismatch."
    )
    boolean errorOnHashMismatch = false;

    /**
     * @deprecated please use --checkHashMismatch WARN instead. Other options are IGNORE and WARN.
     *     <p>This is a hack to support existing users of --noerrorOnHashMismatch. JCommander does
     *     not support setting boolean flags with "--no", so instead we set the default to false and
     *     just ignore anyone who passes --noerrorOnHashMismatch.
     */
    @Deprecated
    @Parameter(names = "--noerrorOnHashMismatch")
    boolean ignored = false;
  }

  /** Converts string flags to paths. Public because JCommander invokes this by reflection. */
  public static class PathFlagConverter implements IStringConverter<Path> {

    @Override
    public Path convert(String text) {
      return FileSystems.getDefault().getPath(text);
    }
  }

  /** Validates that a path exists. Public because JCommander invokes this by reflection. */
  public static class PathExistsValidator implements IValueValidator<Path> {

    @Override
    public void validate(String s, Path path) {
      if (!Files.exists(path)) {
        throw new ParameterException(String.format("%s is not a valid path.", path.toString()));
      }
    }
  }

  /** Validates that a set of paths exist. Public because JCommander invokes this by reflection. */
  public static class AllPathsExistValidator implements IValueValidator<List<Path>> {

    @Override
    public void validate(String s, List<Path> paths) {
      for (Path path : paths) {
        if (!Files.exists(path)) {
          throw new ParameterException(String.format("%s is not a valid path.", path.toString()));
        }
      }
    }
  }

  @VisibleForTesting
  static Multimap<String, Long> getEntriesToOmit(
      Collection<Path> filterZips, Collection<String> filterTypes) throws IOException {
    // Escape filter types to prevent regex abuse
    Set<String> escapedFilterTypes = new LinkedHashSet<>();
    for (String filterType : filterTypes) {
      escapedFilterTypes.add(Pattern.quote(filterType));
    }
    // Match any string that ends with any of the filter file types
    String filterRegex = String.format(".*(%s)$", Joiner.on("|").join(escapedFilterTypes));

    ImmutableSetMultimap.Builder<String, Long> entriesToOmit = ImmutableSetMultimap.builder();
    for (Path filterZip : filterZips) {
      try (ZipReader zip = new ZipReader(filterZip.toFile())) {
        for (ZipFileEntry entry : zip.entries()) {
          if (filterTypes.isEmpty() || entry.getName().matches(filterRegex)) {
            entriesToOmit.put(entry.getName(), entry.getCrc());
          }
        }
      }
    }
    return entriesToOmit.build();
  }

  public static void main(String[] args) throws IOException {
    System.exit(run(args));
  }

  static int run(String[] args) throws IOException {
    Options options = new Options();
    new JCommander(options).parse(args);
    logger.fine(
        String.format(
            "Creating filter from entries of type %s, in zip files %s.",
            options.filterTypes, options.filterZips));

    final Stopwatch timer = Stopwatch.createStarted();
    Multimap<String, Long> entriesToOmit =
        getEntriesToOmit(options.filterZips, options.filterTypes);
    final String explicitFilter =
        options.explicitFilters.isEmpty()
            ? ""
            : String.format(".*(%s).*", Joiner.on("|").join(options.explicitFilters));
    logger.fine(String.format("Filter created in %dms", timer.elapsed(TimeUnit.MILLISECONDS)));

    ImmutableMap.Builder<String, Long> inputEntries = ImmutableMap.builder();
    try (ZipReader input = new ZipReader(options.inputZip.toFile())) {
      for (ZipFileEntry entry : input.entries()) {
        inputEntries.put(entry.getName(), entry.getCrc());
      }
    }

    // TODO(jingwen): Remove --errorOnHashMismatch when Blaze release with --checkHashMismatch
    // is checked in.
    if (options.errorOnHashMismatch) {
      options.hashMismatchCheckMode = HashMismatchCheckMode.ERROR;
    }
    ZipFilterEntryFilter entryFilter =
        new ZipFilterEntryFilter(
            explicitFilter,
            entriesToOmit,
            inputEntries.buildOrThrow(),
            options.hashMismatchCheckMode);

    try (OutputStream out = Files.newOutputStream(options.outputZip);
        ZipCombiner combiner = new ZipCombiner(options.outputMode, entryFilter, out)) {
      combiner.addZip(options.inputZip.toFile());
    }
    logger.fine(String.format("Filtering completed in %dms", timer.elapsed(TimeUnit.MILLISECONDS)));
    return entryFilter.sawErrors() ? 1 : 0;
  }
}
