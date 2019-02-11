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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Handles the 'coverage' command on the Bazel command line.
 *
 * <p>Here follows a brief, partial and probably wrong description of how coverage collection works
 * in Bazel.
 *
 * <p>Coverage is reported by the tests in LCOV format in the files {@code
 * testlogs/PACKAGE/TARGET/coverage.dat} and {@code testlogs/PACKAGE/TARGET/coverage.micro.dat}.
 *
 * <p>To collect coverage, each test execution is wrapped in a script called {@code
 * collect_coverage.sh}. This script sets up the environment of the test to enable coverage
 * collection and determine where the coverage files are written by the coverage runtime(s). It then
 * runs the test. A test may itself run multiple subprocesses and consist of modules written in
 * multiple different languages (with separate coverage runtimes). As such, the wrapper script
 * converts the resulting files to lcov format if necessary, and merges them into a single file.
 *
 * <p>The interposition itself is done by the test strategies, which requires {@code
 * collect_coverage.sh} to be on the inputs of the test. This is accomplished by an implicit
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
 * coverage for with no covered lines. The baseline coverage file for a target is at {@code
 * testlogs/PACKAGE/TARGET/baseline_coverage.dat}. Note that it is also generated for binaries and
 * libraries in addition to tests if you pass the {@code --nobuild_tests_only} flag to Bazel.
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
 * set of inputs of test actions if coverage mode is enabled (otherwise the set of metadata files is
 * empty).
 *
 * <p>Whether or not coverage is being collected is stored in the {@code BuildConfiguration}. This
 * is handy because then we have an easy way to change the test action and the action graph
 * depending on this bit, but it also means that if this bit is flipped, all targets need to be
 * re-analyzed (note that some languages, e.g. C++ require different compiler options to emit code
 * that can collect coverage, which dominates the time required for analysis).
 *
 * <p>The coverage support files are depended on through labels an implicit dependency so that they
 * can be overridden by the invocation policy, which allows them to differ between the different
 * versions of Bazel. Ideally, these differences will be removed, and we standardize on
 *
 * @bazel_tools//tools/coverage.
 *     <p>A partial set of file types that can be encountered in the coverage world:
 *     <ul>
 *       <li><b>{@code .gcno}:</b> Coverage metadata file generated by GCC/Clang.
 *       <li><b>{@code .gcda}:</b> Coverage file generated when a coverage-instrumented binary
 *           compiled by GCC/Clang is run. When combined with the matching {@code .gcno} file, there
 *           is enough data to generate an LCOV file.
 *       <li><b>{@code .instrumented_files}:</b> A text file containing the exec paths of the
 *           instrumented files in a library, binary or test, one in each line. Used to generate the
 *           baseline coverage.
 *       <li><b>{@code coverage.dat}:</b> Coverage data for a single test run.
 *       <li><b>{@code coverage.micro.dat}:</b> Microcoverage data for a single test run.
 *       <li><b>{@code _coverage_report.dat}:</b> Coverage file for a whole Bazel invocation.
 *           Generated in {@code BuildView} in combination with {@code CoverageReportActionFactory}.
 *     </ul>
 *     <p><b>OPEN QUESTIONS:</b>
 *     <ul>
 *       <li>How per-testcase microcoverage data get reported?
 *       <li>How does Jacoco work?
 *     </ul>
 */
@Command(
    name = "coverage",
    builds = true,
    inherits = {TestCommand.class},
    shortDescription = "Generates code coverage report for specified test targets.",
    completion = "label-test",
    help = "resource:coverage.txt",
    allowResidue = true)
public class CoverageCommand extends TestCommand {
  @Override
  protected String commandName() {
    return "coverage";
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {
    super.editOptions(optionsParser);
    try {
      optionsParser.parse(
          PriorityCategory.SOFTWARE_REQUIREMENT,
          "Options required by the coverage command",
          ImmutableList.of("--collect_code_coverage"));
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          "Options suggested for the coverage command",
          ImmutableList.of(TestTimeout.COVERAGE_CMD_TIMEOUT));
    } catch (OptionsParsingException e) {
      // Should never happen.
      throw new IllegalStateException("Unexpected exception", e);
    }
  }
}
