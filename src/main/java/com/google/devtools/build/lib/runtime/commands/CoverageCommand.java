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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.util.Collection;
import java.util.Iterator;
import java.util.Set;
import java.util.SortedSet;

/**
 * Handles the 'coverage' command on the Bazel command line.
 *
 * <p>Here follows a brief, partial and probably wrong description of how coverage collection works
 * in Bazel.
 *
 * <p>Coverage is reported by the tests in LCOV format in the files
 * {@code testlogs/PACKAGE/TARGET/coverage.dat} and
 * {@code testlogs/PACKAGE/TARGET/coverage.micro.dat}.
 *
 * <p>To collect coverage, each test execution is wrapped in a script called
 * {@code collect_coverage.sh}. This script sets up the environment of the test to enable coverage
 * collection and determine where the coverage files are written by the coverage runtime(s). It
 * then runs the test. A test may itself run multiple subprocesses and consist of modules written
 * in multiple different languages (with separate coverage runtimes). As such, the wrapper script
 * converts the resulting files to lcov format if necessary, and merges them into a single file.
 *
 * <p>The interposition itself is done by the test strategies, which requires
 * {@code collect_coverage.sh} to be on the inputs of the test. This is accomplished by an implicit
 * attribute {@code :coverage_support} which is resolved to the value of the configuration flag
 * {@code --coverage_support} (see {@link
 * com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options#coverageSupport}).
 *
 * <p>There are languages for which we do offline instrumentation, meaning that the coverage
 * instrumentation is added at compile time, e.g. for C++, and for others, we do online
 * instrumentation, meaning that coverage instrumentation is added at execution time, e.g. for
 * Javascript.
 *
 * <p>Another core concept is that of <b>baseline coverage</b>. This is essentially the coverage of
 * library, binary, or test if no code in it was run. The problem it solves is that if you want to
 * compute the test coverage for a binary, it is not enough to merge the coverage of all of the
 * tests, because there may be code in the binary that is not linked into any test. Therefore, what
 * we do is to emit a coverage file for every binary, which contains only the files we collect
 * coverage for with no covered lines. The baseline coverage file for a target is at
 * {@code testlogs/PACKAGE/TARGET/baseline_coverage.dat}. Note that it is also generated for
 * binaries and libraries in addition to tests if you pass the {@code --nobuild_tests_only} flag to
 * Bazel.
 *
 * <p>Baseline coverage collection is currently broken.
 *
 * <p>We track two groups of files for coverage collection for each rule: the set of instrumented
 * files and the set of instrumentation metadata files.
 *
 * <p>The set of instrumented files is just that, a set of files to instrument. For online coverage
 * runtimes, this can be used at runtime to decide which files to instrument. It is also used to
 * implement baseline coverage.
 *
 * <p>The set of instrumentation metadata files is the set of extra files a test needs to generate
 * the LCOV files Bazel requires from it. In practice, this consists of runtime-specific files; for
 * example, the gcc compiler emits {@code .gcno} files during compilation. These are added to the
 * set of inputs of test actions if coverage mode is enabled (otherwise the set of metadata files
 * is empty).
 *
 * <p>Whether or not coverage is being collected is stored in the {@code BuildConfiguration}. This
 * is handy because then we have an easy way to change the test action and the action graph
 * depending on this bit, but it also means that if this bit is flipped, all targets need to be
 * re-analyzed (note that some languages, e.g. C++ require different compiler options to emit
 * code that can collect coverage, which dominates the time required for analysis).
 *
 * <p>The coverage support files are depended on through labels in {@code //tools/defaults} and set
 * through command-line options, so that they can be overridden by the invocation policy, which
 * allows them to differ between the different versions of Bazel. Ideally, these differences will
 * be removed, and we standardize on @bazel_tools//tools/coverage.
 *
 * <p>A partial set of file types that can be encountered in the coverage world:
 * <ul>
 *   <li><b>{@code .gcno}:</b> Coverage metadata file generated by GCC/Clang.
 *   <li><b>{@code .gcda}:</b> Coverage file generated when a coverage-instrumented binary compiled
 *   by GCC/Clang is run. When combined with the matching {@code .gcno} file, there is enough data
 *   to generate an LCOV file.
 *   <li><b>{@code .instrumented_files}:</b> A text file containing the exec paths of the
 *   instrumented files in a library, binary or test, one in each line. Used to generate the
 *   baseline coverage.
 *   <li><b>{@code coverage.dat}:</b> Coverage data for a single test run.
 *   <li><b>{@code coverage.micro.dat}:</b> Microcoverage data for a single test run.
 *   <li><b>{@code _coverage_report.dat}:</b> Coverage file for a whole Bazel invocation. Generated
 *   in {@code BuildView} in combination with {@code CoverageReportActionFactory}.
 * </ul>
 *
 * <p><b>OPEN QUESTIONS:</b>
 * <ul>
 *   <li>How per-testcase microcoverage data get reported?
 *   <li>How does Jacoco work?
 * </ul>
 */
@Command(name = "coverage",
         builds = true,
         inherits = { TestCommand.class },
         shortDescription = "Generates code coverage report for specified test targets.",
         completion = "label-test",
         help = "resource:coverage.txt",
         allowResidue = true)
public class CoverageCommand extends TestCommand {
  private boolean wasInterrupted = false;

  @Override
  protected String commandName() {
    return "coverage";
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser)
      throws AbruptExitException {
    super.editOptions(env, optionsParser);
    try {
      optionsParser.parse(OptionPriority.SOFTWARE_REQUIREMENT,
          "Options required by the coverage command",
          ImmutableList.of("--collect_code_coverage"));
      optionsParser.parse(OptionPriority.COMPUTED_DEFAULT,
          "Options suggested for the coverage command",
          ImmutableList.of(TestTimeout.COVERAGE_CMD_TIMEOUT));
      if (!optionsParser.containsExplicitOption("instrumentation_filter")) {
        setDefaultInstrumentationFilter(env, optionsParser);
      }
    } catch (OptionsParsingException e) {
      // Should never happen.
      throw new IllegalStateException("Unexpected exception", e);
    }
  }

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    if (wasInterrupted) {
      wasInterrupted = false;
      env.getReporter().handle(Event.error("Interrupted"));
      return ExitCode.INTERRUPTED;
    }

    return super.exec(env, options);
  }

  /**
   * Method implements a heuristic used to set default value of the
   * --instrumentation_filter option. Following algorithm is used:
   * 1) Identify all test targets on the command line.
   * 2) Expand all test suites into the individual test targets
   * 3) Calculate list of package names containing all test targets above.
   * 4) Replace all "javatests/" substrings in package names with "java/".
   * 5) If two packages reside in the same directory, use filter based on
   *    the parent directory name instead. Doing so significantly simplifies
   *    instrumentation filter in majority of real-life scenarios (in
   *    particular when dealing with my/package/... wildcards).
   * 6) Set --instrumentation_filter default value to instrument everything
   *    in those packages.
   */
  private void setDefaultInstrumentationFilter(CommandEnvironment env,
      OptionsParser optionsProvider)
      throws OptionsParsingException, AbruptExitException {
    try {
      BlazeRuntime runtime = env.getRuntime();
      // Initialize package cache, since it is used by the TargetPatternEvaluator.
      // TODO(bazel-team): Don't allow commands to setup the package cache more than once per build.
      // We'll have to move it earlier in the process to allow this. Possibly: Move it to
      // the command dispatcher and allow commands to annotate "need-packages".
      env.setupPackageCache(optionsProvider, runtime.getDefaultsPackageContent(optionsProvider));

      // Collect all possible test targets. We don't really care whether there will be parsing
      // errors here - they will be reported during actual build.
      TargetPatternEvaluator targetPatternEvaluator = env.newTargetPatternEvaluator();
      Set<Target> testTargets =
          targetPatternEvaluator.parseTargetPatternList(
              env.getReporter(),
              optionsProvider.getResidue(),
              FilteringPolicies.FILTER_TESTS,
              /*keep_going=*/true).getTargets();

      SortedSet<String> packageFilters = Sets.newTreeSet();
      collectInstrumentedPackages(env, testTargets, packageFilters);
      optimizeFilterSet(packageFilters);

      String instrumentationFilter = "//" + Joiner.on(",//").join(packageFilters);
      final String instrumentationFilterOptionName = "instrumentation_filter";
      if (!packageFilters.isEmpty()) {
        env.getReporter().handle(
            Event.info("Using default value for --instrumentation_filter: \""
                + instrumentationFilter + "\"."));

        env.getReporter().handle(Event.info("Override the above default with --"
            + instrumentationFilterOptionName));
        optionsProvider.parse(OptionPriority.COMPUTED_DEFAULT,
                      "Instrumentation filter heuristic",
                      ImmutableList.of("--" + instrumentationFilterOptionName
                                       + "=" + instrumentationFilter));
      }
    } catch (TargetParsingException e) {
      // We can't compute heuristic - just use default filter.
    } catch (InterruptedException e) {
      // We cannot quit now because AbstractCommand does not have the
      // infrastructure to do that. Just set a flag and return from exec() as
      // early as possible. We can do this because there is always an exec()
      // after an editOptions().
      wasInterrupted = true;
    }
  }

  private void collectInstrumentedPackages(CommandEnvironment env,
      Collection<Target> targets, Set<String> packageFilters) throws InterruptedException {
    for (Target target : targets) {
      // Add package-based filters for every test target.
      String prefix = getInstrumentedPrefix(target.getLabel().getPackageName());
      if (!prefix.isEmpty()) {
        packageFilters.add(prefix);
      }
      if (TargetUtils.isTestSuiteRule(target)) {
        AttributeMap attributes = NonconfigurableAttributeMapper.of((Rule) target);
        // We don't need to handle $implicit_tests attribute since we already added
        // test_suite package to the set.
        for (Label label : attributes.get("tests", BuildType.LABEL_LIST)) {
          // Add package-based filters for all tests in the test suite.
          packageFilters.add(getInstrumentedPrefix(label.getPackageName()));
        }
      }
    }
  }

  /**
   * Returns prefix string that should be instrumented for a given package. Input string should
   * be formatted like the output of Label.getPackageName().
   * Generally, package name will be used as such string with two modifications.
   * - "javatests/ directories will be substituted with "java/", since we do
   * not want to instrument java test code. "java/" directories in "test/" will
   * be replaced by the same in "main/".
   * - "/internal", "/public", and "tests/" package suffix will be dropped, since usually we would
   * want to instrument code in the parent package as well
   */
  public static String getInstrumentedPrefix(String packageName) {
    if (packageName.endsWith("/internal")) {
      packageName = packageName.substring(0, packageName.length() - "/internal".length());
    } else if (packageName.endsWith("/public")) {
      packageName = packageName.substring(0, packageName.length() - "/public".length());
    } else if (packageName.endsWith("/tests")) {
      packageName = packageName.substring(0, packageName.length() - "/tests".length());
    }
    return packageName
        .replaceFirst("(?<=^|/)javatests/", "java/")
        .replaceFirst("(?<=^|/)test/java/", "main/java/");
  }

  private static void optimizeFilterSet(SortedSet<String> packageFilters) {
    Iterator<String> iterator = packageFilters.iterator();
    if (iterator.hasNext()) {
      // Find common parent filters to reduce number of filter expressions. In practice this
      // still produces nicely constrained instrumentation filter while making final
      // filter value much more user-friendly - especially in case of /my/package/... wildcards.
      Set<String> parentFilters = Sets.newTreeSet();
      String filterString = iterator.next();
      PathFragment parent = PathFragment.create(filterString).getParentDirectory();
      while (iterator.hasNext()) {
        String current = iterator.next();
        if (parent != null && parent.getPathString().length() > 0
            && !current.startsWith(filterString) && current.startsWith(parent.getPathString())) {
          parentFilters.add(parent.getPathString());
        } else {
          filterString = current;
          parent = PathFragment.create(filterString).getParentDirectory();
        }
      }
      packageFilters.addAll(parentFilters);

      // Optimize away nested filters.
      iterator = packageFilters.iterator();
      String prev = iterator.next();
      while (iterator.hasNext()) {
        String current = iterator.next();
        if (current.startsWith(prev)) {
          iterator.remove();
        } else {
          prev = current;
        }
      }
    }
  }
}
