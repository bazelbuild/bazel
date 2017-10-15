// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Predicate;
import com.google.common.base.Splitter;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A parsed set of configuration filters for a split flag or an output filename.
 *
 * <p>The natural ordering of this class sorts by number of configurations, then by highest required
 * API version, if any, then by other specifiers (case-insensitive), with ties broken by the
 * filename or split flag originally used to create the instance (case-sensitive).
 *
 * <p>This has the following useful property:<br/>
 * Given two sets of {@link SplitConfigurationFilter}s, one from the input split flags, and
 * one from aapt's outputs... Each member of the output set can be matched to the greatest member
 * of the input set for which {@code input.matchesFilterFromFilename(output)} is true.
 */
final class SplitConfigurationFilter implements Comparable<SplitConfigurationFilter> {

  /**
   * Finds a mapping from filename suffixes to the split flags which could have spawned them.
   *
   * @param filenames The suffixes of the original apk filenames output by aapt, not including the
   *     underscore used to set it off from the base filename or the base filename itself.
   * @param splitFlags The split flags originally passed to aapt.
   * @return A map whose keys are the filenames from {@code filenames} and whose values are
   *     predictable filenames based on the split flags - that is, the commas present in the input
   *     have been replaced with underscores.
   * @throws UnrecognizedSplitException if any of the inputs are unused or could not be matched
   */
  static Map<String, String> mapFilenamesToSplitFlags(
      Iterable<String> filenames, Iterable<String> splitFlags) throws UnrecognizedSplitsException {
    TreeSet<SplitConfigurationFilter> filenameFilters = new TreeSet<>();
    for (String filename : filenames) {
      filenameFilters.add(SplitConfigurationFilter.fromFilenameSuffix(filename));
    }
    TreeSet<SplitConfigurationFilter> flagFilters = new TreeSet<>();
    for (String splitFlag : splitFlags) {
      flagFilters.add(SplitConfigurationFilter.fromSplitFlag(splitFlag));
    }
    ImmutableMap.Builder<String, String> result = ImmutableMap.builder();
    List<String> unidentifiedFilenames = new ArrayList<>();
    for (SplitConfigurationFilter filenameFilter : filenameFilters) {
      Optional<SplitConfigurationFilter> matched =
          Iterables.tryFind(flagFilters, new MatchesFilterFromFilename(filenameFilter));
      if (matched.isPresent()) {
        result.put(filenameFilter.filename, matched.get().filename);
        flagFilters.remove(matched.get());
      } else {
        unidentifiedFilenames.add(filenameFilter.filename);
      }
    }
    if (!(unidentifiedFilenames.isEmpty() && flagFilters.isEmpty())) {
      ImmutableList.Builder<String> unidentifiedFlags = ImmutableList.builder();
      for (SplitConfigurationFilter flagFilter : flagFilters) {
        unidentifiedFlags.add(flagFilter.filename);
      }
      throw new UnrecognizedSplitsException(
          unidentifiedFlags.build(), unidentifiedFilenames, result.build());
    }
    return result.build();
  }

  /**
   * Exception thrown when mapFilenamesToSplitFlags fails to find matches for all elements of both
   * input sets.
   */
  static final class UnrecognizedSplitsException extends Exception {
    private final ImmutableList<String> unidentifiedSplits;
    private final ImmutableList<String> unidentifiedFilenames;
    private final ImmutableMap<String, String> identifiedSplits;

    UnrecognizedSplitsException(
        Iterable<String> unidentifiedSplits,
        Iterable<String> unidentifiedFilenames,
        Map<String, String> identifiedSplits) {
      super(
          "Could not find matching filenames for these split flags:\n"
              + Joiner.on("\n").join(unidentifiedSplits)
              + "\nnor matching split flags for these filenames:\n"
              + Joiner.on(", ").join(unidentifiedFilenames)
              + "\nFound these (filename => split flag) matches though:\n"
              + Joiner.on("\n").withKeyValueSeparator(" => ").join(identifiedSplits));
      this.unidentifiedSplits = ImmutableList.copyOf(unidentifiedSplits);
      this.unidentifiedFilenames = ImmutableList.copyOf(unidentifiedFilenames);
      this.identifiedSplits = ImmutableMap.copyOf(identifiedSplits);
    }

    /** Returns the list of split flags which did not find a match. */
    ImmutableList<String> getUnidentifiedSplits() {
      return unidentifiedSplits;
    }

    /** Returns the list of filename suffixes which did not find a match. */
    ImmutableList<String> getUnidentifiedFilenames() {
      return unidentifiedFilenames;
    }

    /** Returns the mapping from filename suffix to split flag for splits that did match. */
    ImmutableMap<String, String> getIdentifiedSplits() {
      return identifiedSplits;
    }
  }

  /** Generates a SplitConfigurationFilter from a split flag. */
  static SplitConfigurationFilter fromSplitFlag(String flag) {
    return SplitConfigurationFilter.fromFilenameSuffix(flag.replace(',', '_'));
  }

  /** Generates a SplitConfigurationFilter from the suffix of a split generated by aapt. */
  static SplitConfigurationFilter fromFilenameSuffix(String suffix) {
    ImmutableSortedSet.Builder<ResourceConfiguration> configs = ImmutableSortedSet.reverseOrder();
    for (String configuration : Splitter.on('_').split(suffix)) {
      configs.add(ResourceConfiguration.fromString(configuration));
    }
    return new SplitConfigurationFilter(suffix, configs.build());
  }

  /**
   * The suffix to be appended to the output package for this split configuration.
   *
   * <p>When created with {@link fromFilenameSuffix}, this will be the original filename from aapt;
   * when created with {@link fromSplitFlag}, this will be the filename to rename to.
   */
  private final String filename;

  /**
   * A set of resource configurations which will be included in this split, sorted so that the
   * configs with the highest API versions come first.
   *
   * <p>It's okay for this to collapse duplicates, because aapt forbids duplicate resource
   * configurations across all splits in the same invocation anyway.
   */
  private final ImmutableSortedSet<ResourceConfiguration> configs;

  private SplitConfigurationFilter(
      String filename, ImmutableSortedSet<ResourceConfiguration> configs) {
    this.filename = filename;
    this.configs = configs;
  }

  /**
   * Checks if the {@code other} split configuration filter could have been produced as a filename
   * by aapt based on this configuration filter being passed as a split flag.
   *
   * <p>This means that there must be a one-to-one mapping from each configuration in this filter to
   * a configuration in the {@code other} filter such that the non-API-version specifiers of the two
   * configurations match and the API version specifier of the {@code other} filter's configuration
   * is greater than or equal to the API version specifier of this filter's configuration.
   *
   * <p>Order of whole configurations doesn't matter, as aapt will reorder the configurations
   * according to complicated internal logic (yes, logic even more complicated than this!).
   *
   * <p>Care is needed with API version specifiers because aapt may add or change minimum
   * API version specifiers to configurations according to whether they had specifiers which are
   * only supported in certain versions of Android. It will only ever increase the minimum version
   * or leave it the same.
   *
   * <p>The other (non-wildcard) specifiers should be case-insensitive identical, including order;
   * aapt will not allow parts of a single configuration to be parsed out of order.
   *
   * @see ResourceConfiguration#matchesConfigurationFromFilename(ResourceConfiguration)
   */
  boolean matchesFilterFromFilename(SplitConfigurationFilter filenameFilter) {
    if (filenameFilter.configs.size() != this.configs.size()) {
      return false;
    }

    List<ResourceConfiguration> unmatchedConfigs = new ArrayList<>(this.configs);
    for (ResourceConfiguration filenameConfig : filenameFilter.configs) {
      Optional<ResourceConfiguration> matched =
          Iterables.tryFind(
              unmatchedConfigs,
              new ResourceConfiguration.MatchesConfigurationFromFilename(filenameConfig));
      if (!matched.isPresent()) {
        return false;
      }
      unmatchedConfigs.remove(matched.get());
    }
    return true;
  }

  static final class MatchesFilterFromFilename implements Predicate<SplitConfigurationFilter> {
    private final SplitConfigurationFilter filenameFilter;

    MatchesFilterFromFilename(SplitConfigurationFilter filenameFilter) {
      this.filenameFilter = filenameFilter;
    }

    @Override
    public boolean apply(SplitConfigurationFilter flagFilter) {
      return flagFilter.matchesFilterFromFilename(filenameFilter);
    }
  }

  private static final Ordering<Iterable<ResourceConfiguration>> CONFIG_LEXICOGRAPHICAL =
      Ordering.natural().lexicographical();

  @Override
  public int compareTo(SplitConfigurationFilter other) {
    return ComparisonChain.start()
        .compare(this.configs.size(), other.configs.size())
        .compare(this.configs, other.configs, CONFIG_LEXICOGRAPHICAL)
        .compare(this.filename, other.filename)
        .result();
  }

  @Override
  public int hashCode() {
    return Objects.hash(configs, filename);
  }

  @Override
  public boolean equals(Object object) {
    if (object instanceof SplitConfigurationFilter) {
      SplitConfigurationFilter other = (SplitConfigurationFilter) object;
      // the configs are derived from the filename, so we can be assured they are equal if the
      // filenames are.
      return Objects.equals(this.filename, other.filename);
    }
    return false;
  }

  @Override
  public String toString() {
    return "SplitConfigurationFilter{" + filename + "}";
  }

  /**
   * An individual set of configuration specifiers, for the purposes of split name parsing.
   *
   * <p>The natural ordering of this class sorts by required API version, if any, then by other
   * specifiers.
   *
   * <p>This has the following useful property:<br/>
   * Given two sets of {@link ResourceConfiguration}s, one from an input split flag, and
   * one from aapt's output... Each member of the output set can be matched to the greatest member
   * of the input set for which {@code input.matchesConfigurationFromFilename(output)} is true.
   */
  static final class ResourceConfiguration implements Comparable<ResourceConfiguration> {
    /**
     * Pattern to match wildcard parts ("any"), which can be safely ignored - aapt drops them.
     *
     * <p>Matches an 'any' part and the dash following it, or for an 'any' part which is the last
     * specifier, the dash preceding it. In the former case, it must be a full part - that is,
     * preceded by the beginning of the string or a dash, which will not be consumed.
     */
    private static final Pattern WILDCARD_SPECIFIER = Pattern.compile("(?<=^|-)any(?:-|$)|-any$");
    /**
     * Pattern to match the API version and capture the version number.
     *
     * <p>It must always be the last specifier in a config, although it may also be the first if
     * there are no other specifiers.
     */
    private static final Pattern API_VERSION = Pattern.compile("(?:-|^)v(\\d+)$");

    /** Parses a resource configuration into a form that can be compared to other configurations. */
    static ResourceConfiguration fromString(String text) {
      // Case is ignored for resource configurations (aapt lowercases internally),
      // and wildcards can be dropped.
      String cleanSpecifiers =
          WILDCARD_SPECIFIER.matcher(text.toLowerCase(Locale.ENGLISH)).replaceAll("");
      Matcher apiVersionMatcher = API_VERSION.matcher(cleanSpecifiers);
      if (apiVersionMatcher.find()) {
        return new ResourceConfiguration(
            cleanSpecifiers.substring(0, apiVersionMatcher.start()),
            Integer.parseInt(apiVersionMatcher.group(1)));
      } else {
        return new ResourceConfiguration(cleanSpecifiers, 0);
      }
    }

    /** The specifiers for this resource configuration, besides API version, in lowercase. */
    private final String specifiers;

    /** The API version, or 0 to indicate that no API version was present in the original config. */
    private final int apiVersion;

    private ResourceConfiguration(String specifiers, int apiVersion) {
      this.specifiers = specifiers;
      this.apiVersion = apiVersion;
    }

    /**
     * Checks that the {@code other} configuration could be a filename generated from this one.
     *
     * @see SplitConfigurationFilter#matchesFilterFromFilename(SplitConfigurationFilter)
     */
    boolean matchesConfigurationFromFilename(ResourceConfiguration other) {
      return Objects.equals(other.specifiers, this.specifiers)
          && other.apiVersion >= this.apiVersion;
    }

    static final class MatchesConfigurationFromFilename
        implements Predicate<ResourceConfiguration> {
      private final ResourceConfiguration filenameConfig;

      MatchesConfigurationFromFilename(ResourceConfiguration filenameConfig) {
        this.filenameConfig = filenameConfig;
      }

      @Override
      public boolean apply(ResourceConfiguration flagConfig) {
        return flagConfig.matchesConfigurationFromFilename(filenameConfig);
      }
    }

    @Override
    public int compareTo(ResourceConfiguration other) {
      return ComparisonChain.start()
          .compare(this.apiVersion, other.apiVersion)
          .compare(this.specifiers, other.specifiers)
          .result();
    }

    @Override
    public int hashCode() {
      return Objects.hash(specifiers, apiVersion);
    }

    @Override
    public boolean equals(Object object) {
      if (object instanceof ResourceConfiguration) {
        ResourceConfiguration other = (ResourceConfiguration) object;
        return Objects.equals(this.specifiers, other.specifiers)
            && this.apiVersion == other.apiVersion;
      }
      return false;
    }

    @Override
    public String toString() {
      return "ResourceConfiguration{" + specifiers + "-v" + Integer.toString(apiVersion) + "}";
    }
  }
}
