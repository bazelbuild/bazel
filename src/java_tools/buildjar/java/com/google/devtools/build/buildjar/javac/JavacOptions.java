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
import java.util.Iterator;
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
    return opt.startsWith("-Werror:")
        || opt.startsWith("-Xep")
        // TODO(b/36228287): delete this once the migration to -XepDisableAllChecks is complete
        || opt.equals("-extra_checks")
        || opt.startsWith("-extra_checks:");
  }

  /**
   * Interface to define an option normalizer. For instance, to group all -Xlint: option into one
   * place.
   *
   * <p>For each option, the first option normalized whose {@link #processOption} method returns
   * true stops its parsing and the option is supposed to be added at the end to the normalized list
   * of option with the {@link #normalize(List)} method. Options not handled by a normalizer will be
   * returned as such in the normalized option list.
   */
  public interface JavacOptionNormalizer {
    /**
     * Process an option and return true if the option was handled by this normalizer. {@code
     * remaining} provides an iterator to any remaining options so normalizers that process
     * non-nullary options can also process the options' arguments.
     */
    boolean processOption(String option, Iterator<String> remaining);

    /**
     * Add the normalized versions of the options handled by {@link #processOption} to the {@code
     * normalized} list
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
    private enum BasisXlintSelection {
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
      resetBasisTo(BasisXlintSelection.Empty);
    }

    @Override
    public boolean processOption(String option, Iterator<String> remaining) {
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
   * Normalizer for {@code -source}, {@code -target}, and {@code --release} options. If both {@code
   * -source} and {@code --release} are specified, {@code --release} wins.
   */
  public static class ReleaseOptionNormalizer implements JavacOptionNormalizer {

    private String source;
    private String target;
    private String release;

    @Override
    public boolean processOption(String option, Iterator<String> remaining) {
      switch (option) {
        case "-source":
          if (remaining.hasNext()) {
            source = remaining.next();
            release = null;
          }
          return true;
        case "-target":
          if (remaining.hasNext()) {
            target = remaining.next();
            release = null;
          }
          return true;
        case "--release":
          if (remaining.hasNext()) {
            release = remaining.next();
            source = null;
            target = null;
          }
          return true;
        default: // fall out
      }
      if (option.startsWith("--release=")) {
        release = option.substring("--release=".length());
        source = null;
        target = null;
        return true;
      }
      return false;
    }

    @Override
    public void normalize(List<String> normalized) {
      if (release != null) {
        normalized.add("--release");
        normalized.add(release);
      } else {
        if (source != null) {
          normalized.add("-source");
          normalized.add(source);
        }
        if (target != null) {
          normalized.add("-target");
          normalized.add(target);
        }
      }
    }
  }

  /**
   * Parse an option that starts with {@code -Werror:} into a bunch of werroropts. We silently drop
   * werroropts that would disable any warnings that we turn into errors by default (treating them
   * like invalid werroropts).
   */
  private static final class WErrorOptionNormalizer implements JavacOptionNormalizer {

    private final WerrorCustomOption.Builder builder;

    WErrorOptionNormalizer(ImmutableList<String> warningsAsErrorsDefault) {
      builder = new WerrorCustomOption.Builder(warningsAsErrorsDefault);
    }

    @Override
    public boolean processOption(String option, Iterator<String> remaining) {
      if (option.startsWith("-Werror:")) {
        builder.process(option);
        return true;
      }
      if (option.equals("-Werror")) {
        builder.all();
        return true;
      }
      return false;
    }

    @Override
    public void normalize(List<String> normalized) {
      String flag = builder.build().toString();
      if (!flag.isEmpty()) {
        normalized.add(flag);
      }
    }
  }

  /**
   * Normalizer for {@code -parameters}, which allows the (non-standard) flag {@code
   * -XDnoparameters} to disable it based on which option appears last in the params list.
   */
  static final class ParameterOptionNormalizer implements JavacOptionNormalizer {

    private static final String PARAMETERS = "-parameters";
    private boolean parameters = false;

    @Override
    public boolean processOption(String option, Iterator<String> remaining) {
      switch (option) {
        case "-XDnoparameters":
          parameters = false;
          return true;
        case PARAMETERS:
          parameters = true;
          return true;
        default:
          return false;
      }
    }

    @Override
    public void normalize(List<String> normalized) {
      if (parameters) {
        normalized.add(PARAMETERS);
      }
    }
  }

  private final ImmutableList<JavacOptionNormalizer> normalizers;

  JavacOptions(ImmutableList<JavacOptionNormalizer> normalizers) {
    this.normalizers = normalizers;
  }

  /**
   * Outputs a reasonably normalized javac option list.
   *
   * @param javacopts the raw javac option list to cleanup
   * @return a new cleaned up javac option list
   */
  public List<String> normalize(List<String> javacopts) {
    List<String> normalized = new ArrayList<>();

    Iterator<String> it = javacopts.iterator();
    while (it.hasNext()) {
      String opt = it.next();
      boolean found = false;
      for (JavacOptionNormalizer normalizer : normalizers) {
        if (normalizer.processOption(opt, it)) {
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
   * Outputs a reasonably normalized javac option list.
   *
   * @param javacopts the raw javac option list to cleanup
   * @param normalizers the list of normalizers to apply
   * @return a new cleaned up javac option list
   */
  public static List<String> normalizeOptionsWithNormalizers(
      List<String> javacopts, JavacOptionNormalizer... normalizers) {
    return new JavacOptions(ImmutableList.copyOf(normalizers)).normalize(javacopts);
  }

  /**
   * Creates a {@link JavacOptions} normalizer that will ensure the given set of lint categories are
   * enabled as errors, overriding any user-provided configuration for those options.
   */
  public static JavacOptions createWithWarningsAsErrorsDefault(
      ImmutableList<String> warningsAsErrorsDefault) {
    return new JavacOptions(
        ImmutableList.of(
            new XlintOptionNormalizer(warningsAsErrorsDefault),
            new WErrorOptionNormalizer(warningsAsErrorsDefault),
            new ReleaseOptionNormalizer(),
            new ParameterOptionNormalizer()));
  }
}
