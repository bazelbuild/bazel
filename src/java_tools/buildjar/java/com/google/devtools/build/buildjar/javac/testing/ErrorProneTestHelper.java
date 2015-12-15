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

package com.google.devtools.build.buildjar.javac.testing;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.BlazeJavacMain;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.plugins.errorprone.ErrorPronePlugin;
import com.google.devtools.build.java.bazel.BazelJavaCompiler;
import com.google.errorprone.ErrorProneAnalyzer;
import com.google.errorprone.ErrorProneOptions;
import com.google.errorprone.scanner.Scanner;
import com.google.errorprone.scanner.ScannerSupplier;

import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskEvent.Kind;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.main.Main.Result;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.util.Options;

import org.junit.After;
import org.junit.Before;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaFileObject;

/**
 * Helper class for testing Error Prone plugins and scanners.
 *
 * <p>
 * This class takes care of the set up of the error-prone compiler, and provides methods to perform
 * compilation and assert success/failure results.
 *
 * <p>
 * Subclasses must invoke (typically from their {@link #setUp()} method) either
 * {@link #registerPlugin(BlazeJavaCompilerPlugin)} or {@link #registerScanner(Scanner)} to install
 * the plugin or scanner to be used during compilation.
 *
 * <p>
 * Java source files are kept in memory using {@link #fileManager}, a
 * {@link InMemoryJavaFileManager}. Test cases should use
 * {@link InMemoryJavaFileManager#addSource(String, String...)} to add Java sources under test.
 */
public abstract class ErrorProneTestHelper {

  private BlazeJavaCompilerPlugin registeredPlugin;

  private Context context;
  private BlazeJavacMain compilerMain;
  private Writer out = new StringWriter();

  /**
   * The file manager to used to keep Java sources under test in memory.
   *
   * <p>
   * Tests cases should use {@link InMemoryJavaFileManager#addSource(String, String...)} to add Java
   * sources under test, and should use {@link InMemoryJavaFileManager#takeAvailableSources()} to
   * obtain the collection of {@link JavaFileObject} to pass to
   * {@link #assertCompileSucceeds(java.util.List)} and {@code ...CompileFails...}.
   * Note that {@link InMemoryJavaFileManager#takeAvailableSources()} clears the file manager, and
   * should only be invoked once per test case.
   *
   * <p>
   * The file manager is initialized by this class' {@link #setUp()} method.
   */
  protected InMemoryJavaFileManager fileManager;

  /**
   * The diagnostics collector in which compilation results are available.
   *
   * <p>
   * After invoking compilation, tests may inspect this collector for details of compilation errors.
   *
   * <p>
   * The diagnostics collector is initialized by this class' {@link #setUp()} method.
   */
  protected DiagnosticCollector<JavaFileObject> collector;

  @Before
  public void setUp() throws Exception {
    registeredPlugin = null;

    // Order is important here, since Context can raise exceptions if setting keys twice!
    context = new Context();
    // Disables faulty Zip implementation.  In the compiler this is done in BlazeJavacMain, and
    // is replicated here to align the test environment with the actual compiler.
    Options options = Options.instance(context);
    options.put("useOptimizedZip", "false");

    collector = new DiagnosticCollector<>();
    fileManager = new InMemoryJavaFileManager();
    compilerMain = new BlazeJavacMain(new PrintWriter(out, true),
        ImmutableList.<BlazeJavaCompilerPlugin>of());
  }

  @After
  public void tearDown() throws Exception {
    fileManager.close();
  }

  protected void registerPlugin(BlazeJavaCompilerPlugin plugin) {
    registeredPlugin = plugin;
    compilerMain.preRegister(context, Arrays.asList(plugin));
  }

  protected void registerScanner(ScannerSupplier supplier) {
    registerScanner(supplier.get());
  }

  protected void registerScanner(Scanner scanner) {
    registerPlugin(new ScannerPlugin(scanner));
  }

  protected String getOutput() {
    return out.toString();
  }

  protected Result doCompile(List<JavaFileObject> sources) {
    return doCompile(new String[]{}, sources);
  }

  protected Result doCompile(String[] args, List<JavaFileObject> sources) {
    assertNotNull("Tests must register a plugin or scanner before compilation", registeredPlugin);
    List<String> argList = new ArrayList<>(BazelJavaCompiler.getDefaultJavacopts());
    argList.addAll(Arrays.asList(args));

    List<String> remainingArgs = null;
    try {
      remainingArgs = registeredPlugin.processArgs(argList);
    } catch (InvalidCommandLineException e) {
      fail("Invalid command line argument: " + e.getMessage());
    }
    fileManager.initializeClasspath(ImmutableList.<Path>of());
    return compilerMain.compile(remainingArgs.toArray(new String[remainingArgs.size()]), context,
        fileManager, collector, sources, null);
  }

  protected List<Diagnostic<? extends JavaFileObject>> assertCompileSucceeds(
      List<JavaFileObject> sources) {
    return assertCompileSucceeds(sources, new String[] {});
  }

  protected List<Diagnostic<? extends JavaFileObject>> assertCompileSucceeds(
      List<JavaFileObject> sources, String[] args) {
    Result compilerReturnCode = doCompile(args, sources);
    assertEquals(getOutput(), Result.OK, compilerReturnCode);
    return collector.getDiagnostics();
  }

  protected void assertCompileSucceedsWithoutWarnings(List<JavaFileObject> sources) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics = assertCompileSucceeds(sources);
    assertThat(diagnostics).isEmpty();
  }

  /**
   * Compiles the given sources and asserts that the compile succeeds with exactly one warning,
   * with an error message that contains a substring that matches the given regex, and at the
   * specified exact source line and column.
   */
  protected void assertCompileSucceedsWithOnlyWarningContaining(
      List<JavaFileObject> sources, String expectedErrorRegex, int line, int col) {
    // TODO(bazel-team): remove column from this check
    List<Diagnostic<? extends JavaFileObject>> diagnostics = assertCompileSucceeds(sources);

    assertThat(diagnostics).named("diagnostics").hasSize(1);
    Diagnostic<? extends JavaFileObject> diagnostic = diagnostics.get(0);
    assertWithMessage("Wrong warning")
        .that(diagnostic.getMessage(null))
        .containsMatch(expectedErrorRegex);
    assertEquals("Wrong line number", line, diagnostic.getLineNumber());
    assertEquals("Wrong column number", col, diagnostic.getColumnNumber());
  }

  /**
   * Compiles the given sources and asserts that the compile succeeds and contains a warning
   * with an error message that contains a substring that matches the given regex, and at the
   * specified exact source line and column.
   */
  protected void assertCompileSucceedsWithWarningContaining(
      List<JavaFileObject> sources, String expectedErrorRegex, int line, int col) {
    // TODO(bazel-team): remove column from this check
    List<Diagnostic<? extends JavaFileObject>> diagnostics = assertCompileSucceeds(sources);

    assertThat(diagnostics).named("diagnostics").isNotEmpty();
    Diagnostic<? extends JavaFileObject> closeMatch =
        findClosestMatch(diagnostics, expectedErrorRegex, line, col);

    assertNotNull("Did not find expected warning", closeMatch);
    assertEquals("Wrong line number", line, closeMatch.getLineNumber());
    assertEquals("Wrong column number", col, closeMatch.getColumnNumber());
  }

  /**
   * Returns a diagnostic that matches the regex. If one with the matching line
   * and column is found, returns that one, otherwise anything that matches the
   * regex. If no regex match is found, returns null.
   */
  private Diagnostic<? extends JavaFileObject> findClosestMatch(
      List<Diagnostic<? extends JavaFileObject>> diagnostics,
      String regex, int line, int col) {
    // TODO(bazel-team): remove column from this check
    Diagnostic<? extends JavaFileObject> closeMatch = null;
    Pattern pattern = Pattern.compile(regex);

    for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics) {
      if (pattern.matcher(diagnostic.getMessage(null)).matches()) {
        if (line == diagnostic.getLineNumber() && col == diagnostic.getColumnNumber()) {
          return diagnostic;
        } else {
          closeMatch = diagnostic;
        }
      }
    }
    return closeMatch;
  }

  /**
   * Compiles the given sources and asserts that the compile succeeds and contains no warnings
   * that match the expected regex.
   */
  protected void assertCompileSucceedsWithoutWarningContaining(
      List<JavaFileObject> sources, String expectedErrorRegex) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics = assertCompileSucceeds(sources);

    for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics) {
      assertWithMessage("Should not contain this warning")
          .that(diagnostic.getMessage(null))
          .doesNotContainMatch(expectedErrorRegex);
    }
  }

  /**
   * Compiles the given sources and asserts that the compile fails.
   *
   * @param sources the sources to compile
   * @return the resulting diagnostics, for further inspection
   */
  protected List<Diagnostic<? extends JavaFileObject>> assertCompileFails(
      List<JavaFileObject> sources) {
    return assertCompileFails(new String[]{}, sources);
  }

  protected List<Diagnostic<? extends JavaFileObject>> assertCompileFails(
      String[] args, List<JavaFileObject> sources) {
    Result compilerReturnCode = doCompile(args, sources);
    assertEquals(getOutput(), Result.ERROR, compilerReturnCode);
    return collector.getDiagnostics();
  }

  /**
   * Compiles the given sources and asserts that the compile fails with exactly one compile error.
   *
   * @param sources the sources to compile
   * @return the resulting diagnostics, for further inspection
   */
  protected List<Diagnostic<? extends JavaFileObject>> assertCompileFailsWithOneError(
      List<JavaFileObject> sources) {
    return assertCompileFailsWithOneError(new String[]{}, sources);
  }

  protected List<Diagnostic<? extends JavaFileObject>> assertCompileFailsWithOneError(
      String[] args, List<JavaFileObject> sources) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics = assertCompileFails(args, sources);
    assertEquals("Expected 1 compiler message, found " + diagnostics.size() + ": " + diagnostics, 1,
        diagnostics.size());
    return diagnostics;
  }

  /**
   * Compiles the given sources and asserts that the compile fails with exactly one compile error,
   * which must contain the expected message as a substring.
   */
  protected void assertCompileFailsWithErrorContaining(
      List<JavaFileObject> sources, String errorMsg) {
    assertCompileFailsWithErrorContaining(new String[] {}, sources, errorMsg);
  }

  protected void assertCompileFailsWithErrorContaining(
      String[] args, List<JavaFileObject> sources, String errorMsg) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics =
        assertCompileFailsWithOneError(args, sources);

    assertTrue("Error message should contain " + errorMsg + ", was: "
        + diagnostics.get(0).getMessage(null),
        diagnostics.get(0).getMessage(null).contains(errorMsg));
  }

  /**
   * Compiles the given sources and asserts that the compile fails with exactly one compile error,
   * with a specific, exact error message, and at the specified exact source line and column.
   */
  protected void assertCompileFailsWithError(List<JavaFileObject> sources,
      String expectedError, int line, int col) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics =
        assertCompileFailsWithOneError(sources);

    Diagnostic<? extends JavaFileObject> diagnostic = diagnostics.get(0);
    assertEquals("Wrong error message", expectedError, diagnostic.getMessage(null));
    assertEquals("Wrong line number", line, diagnostic.getLineNumber());
    assertEquals("Wrong column number", col, diagnostic.getColumnNumber());
  }

  /**
   * Compiles the given sources and asserts that the compile fails with exactly one compile error,
   * with an error message that matches the given regex, and at the specified exact source line and
   * column.
   */
  protected void assertCompileFailsWithErrorMatching(
      List<JavaFileObject> sources, String expectedErrorRegex, int line, int col) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics =
        assertCompileFailsWithOneError(sources);

    Diagnostic<? extends JavaFileObject> diagnostic = diagnostics.get(0);
    assertWithMessage("Wrong error message")
        .that(diagnostic.getMessage(null))
        .matches(expectedErrorRegex);
    assertEquals("Wrong line number", line, diagnostic.getLineNumber());
    assertEquals("Wrong column number", col, diagnostic.getColumnNumber());
  }

  /**
   * Compiles the given sources and asserts that the compile fails with exactly one compile error,
   * with an error message that contains a substring that matches the given regex, and at the
   * specified exact source line and column.
   */
  protected void assertCompileFailsWithErrorContaining(
      List<JavaFileObject> sources, String expectedErrorRegex, int line, int col) {
    List<Diagnostic<? extends JavaFileObject>> diagnostics =
        assertCompileFailsWithOneError(sources);

    Diagnostic<? extends JavaFileObject> diagnostic = diagnostics.get(0);
    assertWithMessage("Wrong error message")
        .that(diagnostic.getMessage(null))
        .containsMatch(expectedErrorRegex);
    assertEquals("Wrong line number", line, diagnostic.getLineNumber());
    assertEquals("Wrong column number", col, diagnostic.getColumnNumber());
  }

  /**
   * A slightly stripped down version of ErrorPronePlugin, hard-wired to use a specific
   * {@link Scanner}.
   *
   * <p>
   * This plugin is installed by {@link ErrorProneTestHelper#registerScanner(Scanner)}, and is
   * intended to be used by unit tests of a specific scanner (as opposed to ErrorProneScanner).
   */
  private final class ScannerPlugin extends BlazeJavaCompilerPlugin {

    private final Scanner scanner;

    private boolean isInitialized = false;
    private ErrorProneAnalyzer errorProneAnalyzer;

    ScannerPlugin(Scanner scanner) {
      this.scanner = scanner;
    }

    @Override
    public void init(Context context, Log log, JavaCompiler compiler) {
      Preconditions.checkState(!isInitialized);
      super.init(context, log, compiler);
      ErrorPronePlugin.setupMessageBundle(context);
      errorProneAnalyzer =
          ErrorProneAnalyzer.create(scanner).init(context, ErrorProneOptions.empty());
      isInitialized = true;
    }

    /**
     * Run Error Prone analysis after performing dataflow checks.
     */
    @Override
    public void postFlow(Env<AttrContext> env) {
      checkState(isInitialized);
      errorProneAnalyzer.finished(new TaskEvent(Kind.ANALYZE, env.toplevel, env.enclClass.sym));
    }
  }
}
