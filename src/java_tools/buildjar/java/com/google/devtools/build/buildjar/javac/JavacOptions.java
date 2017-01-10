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

package com.google.devtools.build.buildjar.javac;

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Preprocess javac -Xlint options. We also need to make the different versions of javac treat
 * -Xlint options uniformly.
 *
 * <p>Some versions of javac now process the -Xlint options without allowing later options to
 * override earlier ones on the command line. For example, {@code -Xlint:All -Xlint:None} results in
 * all warnings being enabled.
 *
 * <p>This class preprocesses the -Xlint options within the javac options to achieve a command line
 * that is sensitive to ordering. That is, with this preprocessing step, {@code -Xlint:all
 * -Xlint:none} results in no warnings being enabled.
 */
public final class JavacOptions {

  /** Returns an immutable list containing all the non-Bazel specific Javac flags. */
  public static ImmutableList<String> removeBazelSpecificFlags(String[] javacopts) {
    return removeBazelSpecificFlags(Arrays.asList(javacopts));
  }

  /** Returns an immutable list containing all the non-Bazel specific Javac flags. */
  public static ImmutableList<String> removeBazelSpecificFlags(Iterable<String> javacopts) {
    return filterJavacopts(javacopts).standardJavacopts();
  }

  /** A collection of javac flags, divided into Bazel-specific and standard options. */
  @AutoValue
  public abstract static class FilteredJavacopts {
    /** Bazel-specific javac flags, e.g. Error Prone's -Xep: flags. */
    public abstract ImmutableList<String> bazelJavacopts();

    /** Standard javac flags. */
    public abstract ImmutableList<String> standardJavacopts();

    /** Creates a {@link FilteredJavacopts}. */
    public static FilteredJavacopts create(
        ImmutableList<String> bazelJavacopts, ImmutableList<String> standardJavacopts) {
      return new AutoValue_JavacOptions_FilteredJavacopts(bazelJavacopts, standardJavacopts);
    }
  }

  /** Filters a list of javac flags into Bazel-specific and standard flags. */
  public static FilteredJavacopts filterJavacopts(Iterable<String> javacopts) {
    ImmutableList.Builder<String> bazelJavacopts = ImmutableList.builder();
    ImmutableList.Builder<String> standardJavacopts = ImmutableList.builder();
    for (String opt : javacopts) {
      if (isBazelSpecificFlag(opt)) {
        bazelJavacopts.add(opt);
      } else {
        standardJavacopts.add(opt);
      }
    }
    return FilteredJavacopts.create(bazelJavacopts.build(), standardJavacopts.build());
  }

  private static boolean isBazelSpecificFlag(String opt) {
    return opt.startsWith("-Werror:") || opt.startsWith("-Xep");
  }

  private static final XlintOptionNormalizer XLINT_OPTION_NORMALIZER = new XlintOptionNormalizer();

  /**
   * Interface to define an option normalizer. For instance, to group all -Xlint: option into one
   * place.
   *
   * <p>All normalizers used by the JavacOptions class will be started by calling the {@link
   * #start()} method when starting the parsing of a list of option. For each option, the first
   * option normalized whose {@link #processOption(String)} method returns true stops its parsing
   * and the option is supposed to be added at the end to the normalized list of option with the
   * {@link #normalize(List)} method. Options not handled by a normalizer will be returned as such
   * in the normalized option list.
   */
  public static interface JavacOptionNormalizer {
    /** Resets the state of the normalizer to start a new option parsing. */
    void start();

    /** Process an option and return true if the option was handled by this normalizer. */
    boolean processOption(String option);

    /**
     * Add the normalized versions of the options handled by {@link #processOption(String)} to the
     * {@code normalized} list
     */
    void normalize(List<String> normalized);
  }

  /**
   * Parse an option that starts with {@code -Xlint:} into a bunch of xlintopts. We silently drop
   * xlintopts that would disable any warnings that we turn into errors by default (treating them
   * like invalid xlintopts). It also parse -nowarn option as -Xlint:none.
   */
  public static final class XlintOptionNormalizer implements JavacOptionNormalizer {

    private static final Joiner COMMA_MINUS_JOINER = Joiner.on(",-");
    private static final Joiner COMMA_JOINER = Joiner.on(",");

    /**
     * This type models a starting selection from which lint options can be added or removed. E.g.,
     * {@code -Xlint} indicates we start with the set of recommended checks enabled, and {@code
     * -Xlint:none} means we start without any checks enabled.
     */
    private static enum BasisXlintSelection {
      /** {@code -Xlint:none} */
      None,
      /** {@code -Xlint:all} */
      All,
      /** {@code -Xlint} */
      Recommended,
      /** Nothing specified; default} */
      Empty
    }

    private final ImmutableList<String> enforcedXlints;
    private final Set<String> xlintPlus;
    private final Set<String> xlintMinus = new LinkedHashSet<>();
    private BasisXlintSelection xlintBasis = BasisXlintSelection.Empty;

    public XlintOptionNormalizer() {
      this(ImmutableList.<String>of());
    }

    public XlintOptionNormalizer(ImmutableList<String> enforcedXlints) {
      this.enforcedXlints = enforcedXlints;
      xlintPlus = new LinkedHashSet<>(enforcedXlints);
    }

    @Override
    public void start() {
      resetBasisTo(BasisXlintSelection.Empty);
    }

    @Override
    public boolean processOption(String option) {
      if (option.equals("-nowarn")) {
        // It is equivalent to -Xlint:none
        resetBasisTo(BasisXlintSelection.None);
        return true;
      } else if (option.equals("-Xlint")) {
        resetBasisTo(BasisXlintSelection.Recommended);
        return true;
      } else if (option.startsWith("-Xlint")) {
        for (String arg : option.substring("-Xlint:".length()).split(",", -1)) {
          if (arg.equals("all") || arg.isEmpty()) {
            resetBasisTo(BasisXlintSelection.All);
          } else if (arg.equals("none")) {
            resetBasisTo(BasisXlintSelection.None);
          } else if (arg.startsWith("-")) {
            arg = arg.substring("-".length());
            if (!enforcedXlints.contains(arg)) {
              xlintPlus.remove(arg);
              if (xlintBasis != BasisXlintSelection.None) {
                xlintMinus.add(arg);
              }
            }
          } else { // not a '-' prefix
            xlintMinus.remove(arg);
            if (xlintBasis != BasisXlintSelection.All) {
              xlintPlus.add(arg);
            }
          }
        }
        return true;
      }
      return false;
    }

    @Override
    public void normalize(List<String> normalized) {
      switch (xlintBasis) {
        case Recommended:
          normalized.add("-Xlint");
          break;
        case All:
          normalized.add("-Xlint:all");
          break;
        case None:
          if (xlintPlus.isEmpty()) {
            /*
             * This should never happen with warnings as errors. The plus set should always contain
             * at least the warnings in warningsAsErrors.
             */
            normalized.add("-Xlint:none");
          }
          break;
        default:
          break;
      }
      if (xlintBasis != BasisXlintSelection.All && !xlintPlus.isEmpty()) {
        normalized.add("-Xlint:" + COMMA_JOINER.join(xlintPlus));
      }
      if (xlintBasis != BasisXlintSelection.None && !xlintMinus.isEmpty()) {
        normalized.add("-Xlint:-" + COMMA_MINUS_JOINER.join(xlintMinus));
      }
    }

    private void resetBasisTo(BasisXlintSelection selection) {
      xlintBasis = selection;
      xlintPlus.clear();
      xlintMinus.clear();
      if (selection != BasisXlintSelection.All) {
        xlintPlus.addAll(enforcedXlints);
      }
    }
  }

  /**
   * Outputs a reasonably normalized javac option list.
   *
   * @param javacopts the raw javac option list to cleanup
   * @param normalizers the list of normalizers to apply
   * @return a new cleaned up javac option list
   */
  public static List<String> normalizeOptionsWithNormalizers(
      List<String> javacopts, JavacOptionNormalizer... normalizers) {
    List<String> normalized = new ArrayList<>();

    for (JavacOptionNormalizer normalizer : normalizers) {
      normalizer.start();
    }

    for (String opt : javacopts) {
      boolean found = false;
      for (JavacOptionNormalizer normalizer : normalizers) {
        if (normalizer.processOption(opt)) {
          found = true;
          break;
        }
      }
      if (!found) {
        normalized.add(opt);
      }
    }

    for (JavacOptionNormalizer normalizer : normalizers) {
      normalizer.normalize(normalized);
    }
    return normalized;
  }

  /**
   * A wrapper around {@ref #normalizeOptionsWithNormalizers(List, JavacOptionNormalizer...)} to use
   * {@link XlintOptionNormalizer} as default normalizer.
   *
   * <p>The -Xlint option list has up to one each of a -Xlint* basis flag followed by a
   * -Xlint:xxx,yyy,zzz add flag followed by a -Xlint:-xxx,-yyy,-zzz minus flag.
   */
  public static List<String> normalizeOptions(List<String> javacopts) {
    return normalizeOptionsWithNormalizers(javacopts, XLINT_OPTION_NORMALIZER);
  }
}
