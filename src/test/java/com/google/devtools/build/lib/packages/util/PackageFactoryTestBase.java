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
package com.google.devtools.build.lib.packages.util;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.GlobCache;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.PackageValidator;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import org.junit.Before;

/**
 * Base class for PackageFactory tests.
 */
public abstract class PackageFactoryTestBase {

  protected Scratch scratch;
  protected EventCollectionApparatus events = new EventCollectionApparatus();
  protected DummyPackageValidator dummyPackageValidator = new DummyPackageValidator();
  protected PackageFactoryApparatus packages =
      new PackageFactoryApparatus(
          events.reporter(), getEnvironmentExtensions(), dummyPackageValidator);
  protected Root root;

  protected com.google.devtools.build.lib.packages.Package expectEvalSuccess(String... content)
      throws InterruptedException, IOException, NoSuchPackageException {
    Path file = scratch.file("pkg/BUILD", content);
    Package pkg = packages.eval("pkg", RootedPath.toRootedPath(root, file));
    assertThat(pkg.containsErrors()).isFalse();
    return pkg;
  }

  protected void expectEvalError(String expectedError, String... content) throws Exception {
    events.setFailFast(false);
    Path file = scratch.file("pkg/BUILD", content);
    Package pkg = packages.eval("pkg", RootedPath.toRootedPath(root, file));
    assertWithMessage("Expected evaluation error, but none was not reported")
        .that(pkg.containsErrors())
        .isTrue();
    events.assertContainsError(expectedError);
  }

  protected abstract List<EnvironmentExtension> getEnvironmentExtensions();

  protected Path throwOnReaddir = null;

  protected static AttributeMap attributes(Rule rule) {
    return RawAttributeMapper.of(rule);
  }

  protected static void assertOutputFileForRule(Package pkg, Collection<String> outNames, Rule rule)
      throws Exception {
    for (String outName : outNames) {
      OutputFile out = (OutputFile) pkg.getTarget(outName);
      assertThat(rule.getOutputFiles()).contains(out);
      assertThat(out.getGeneratingRule()).isSameInstanceAs(rule);
      assertThat(out.getName()).isEqualTo(outName);
      assertThat(out.getTargetKind()).isEqualTo("generated file");
    }
    assertThat(rule.getOutputFiles()).hasSize(outNames.size());
  }

  protected static void assertEvaluates(Package pkg, List<String> expected, String... include)
      throws Exception {
    assertEvaluates(pkg, expected, ImmutableList.copyOf(include), Collections.<String>emptyList());
  }

  protected static void assertEvaluates(
      Package pkg, List<String> expected, List<String> include, List<String> exclude)
      throws Exception {
    GlobCache globCache =
        new GlobCache(
            pkg.getFilename().asPath().getParentDirectory(),
            pkg.getPackageIdentifier(),
            ImmutableSet.of(),
            PackageFactoryApparatus.createEmptyLocator(),
            null,
            TestUtils.getPool(),
            -1);
    assertThat(globCache.globUnsorted(include, exclude, false, true))
        .containsExactlyElementsIn(expected);
  }

  @Before
  public final void initializeFileSystem() throws Exception {
    FileSystem fs =
        new InMemoryFileSystem() {
          @Override
          public Collection<Dirent> readdir(Path path, boolean followSymlinks) throws IOException {
            if (path.equals(throwOnReaddir)) {
              throw new FileNotFoundException(path.getPathString());
            }
            return super.readdir(path, followSymlinks);
          }
        };
    Path tmpPath = fs.getPath("/");
    scratch = new Scratch(tmpPath);
    root = Root.fromPath(scratch.dir("/"));
  }

  protected Path emptyBuildFile(String packageName) {
    return emptyFile(getPathPrefix() + "/" + packageName + "/BUILD");
  }

  protected Path emptyFile(String path) {
    try {
      return scratch.file(path);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  protected boolean isValidPackageName(String packageName) throws Exception {
    // Write a license decl just in case it's a third_party package:
    Path buildFile = scratch.file(
        getPathPrefix() + "/" + packageName + "/BUILD", "licenses(['notice'])");
    Package pkg = packages.createPackage(packageName, RootedPath.toRootedPath(root, buildFile));
    return !pkg.containsErrors();
  }

  /********************************************************************
   *                                                                  *
   *              Test "glob" function in build language              *
   *                                                                  *
   ********************************************************************/
  protected void assertGlobFails(String globCallExpression, String expectedError) throws Exception {
    Package pkg = buildPackageWithGlob(globCallExpression);

    events.assertContainsError(expectedError);
    assertThat(pkg.containsErrors()).isTrue();
  }

  private Package buildPackageWithGlob(String globCallExpression) throws Exception {
    scratch.deleteFile("/dummypackage/BUILD");
    Path file = scratch.file("/dummypackage/BUILD", "x = " + globCallExpression);
    return packages.eval("dummypackage", RootedPath.toRootedPath(root, file));
  }

  private List<Pair<String, Boolean>> createGlobCacheKeys(
      List<String> expressions, boolean excludeDirs) {
    List<Pair<String, Boolean>> keys = Lists.newArrayListWithCapacity(expressions.size());
    for (String expression : expressions) {
      keys.add(Pair.of(expression, excludeDirs));
    }

    return keys;
  }

  /**
   * Test globbing in the context of a package, using the build language.
   * We use the specially setup "globs" test package and the files beneath it.
   * @param result the expected list of filenames that match the glob
   * @param includes an include pattern for the glob
   * @param excludes an exclude pattern for the glob
   * @param excludeDirs an exclude_directories flag for the glob
   * @throws Exception if the glob doesn't match the expected result.
   */
  protected void assertGlobMatches(
      List<String> result, List<String> includes, List<String> excludes, boolean excludeDirs)
      throws Exception {

    Pair<Package, GlobCache> evaluated =
        evaluateGlob(
            includes,
            excludes,
            excludeDirs,
            Starlark.format("(result == sorted(%r)) or fail('incorrect glob result')", result));

    Package pkg = evaluated.first;
    GlobCache globCache = evaluated.second;

    // Ensure all of the patterns are recorded against this package:
    assertThat(globCache.getKeySet().containsAll(createGlobCacheKeys(includes, excludeDirs)))
        .isTrue();
    assertThat(pkg.containsErrors()).isFalse();
  }

  /**
   * Evaluate a glob() call against a test directory and BUILD code to process the results.
   * @param includes a list of glob patterns; glob will include these files.
   * @param excludes a list of glob patterns to exclude even if previously included.
   * @param excludeDirs true if directories should be excluded from the match.
   * @param resultAssertion code in the BUILD language that can access the variable result,
   * to which the result of the glob will be bound, and that may contain an assertion on it.
   * @return a Package and a GlobCache.
   * @throws Exception if the processResult code causes a failure.
   */
  private Pair<Package, GlobCache> evaluateGlob(
      List<String> includes, List<String> excludes, boolean excludeDirs, String resultAssertion)
      throws Exception {
    Path globsDir = scratch.dir("/globs");
    globsDir.getChild("subdir").createDirectory();
    for (String file : ImmutableList.of("Wombat1.java", "Wombat2.java", "subdir/Wombat3.java")) {
      FileSystemUtils.createEmptyFile(globsDir.getRelative(file));
    }
    Path file =
        scratch.file(
            "/globs/BUILD",
            Starlark.format(
                "result = glob(%r, exclude=%r, exclude_directories=%r)",
                includes, excludes, excludeDirs ? 1 : 0),
            resultAssertion);

    return packages.evalAndReturnGlobCache(
        "globs", RootedPath.toRootedPath(root, file), packages.parse(file));
  }

  protected void assertGlobProducesError(String pattern, boolean errorExpected) throws Exception {
    events.setFailFast(false);
    Package pkg =
        evaluateGlob(ImmutableList.of(pattern), Collections.<String>emptyList(), false, "").first;
    assertThat(pkg.containsErrors()).isEqualTo(errorExpected);
    boolean foundError = false;
    for (Event event : events.collector()) {
      if (event.getMessage().contains("glob")) {
        if (!errorExpected) {
          fail("error not expected for glob pattern " + pattern + ", but got: " + event);
          return;
        }
        foundError = errorExpected;
        break;
      }
    }
    assertThat(foundError).isEqualTo(errorExpected);
  }

  /** Runnable that asks for parsing of build file and synchronizes it with
   * ErrorReporter. It consumes log messages from PackageFactory to release
   * first semaphore when parsing is started and waits for second semaphore
   * before it ends.
   */
  protected class ParsingTracker extends Handler implements Runnable {
    private final Semaphore parsingStarted;
    private final Semaphore errorReported;
    private final ExtendedEventHandler eventHandler;
    private boolean first = true;
    private boolean parsedOK;

    public ParsingTracker(Semaphore first, Semaphore second, ExtendedEventHandler eventHandler) {
      this.eventHandler = eventHandler;
      parsingStarted = first;
      errorReported = second;
    }

    @Override
    public void run() {
      try {
        Path buildFile =
            scratch.file(
                getPathPrefix() + "/isolated/BUILD",
                "# -*- python -*-",
                "",
                "java_library(name = 'mylib',",
                "  srcs = 'java/A.java')");
        packages.createPackage(
            PackageIdentifier.createInMainRepo("isolated"),
            RootedPath.toRootedPath(root, buildFile),
            eventHandler);
        parsedOK = true;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    }

    public boolean hasParsed() {
      return parsedOK;
    }

    @Override
    public void close() throws SecurityException {}

    @Override
    public void flush() {}

    @Override
    public void publish(LogRecord record) {
      if (!record.getMessage().contains("isolated")) {
        return;
      }

      if (first) {
        parsingStarted.release();
        first = false;
      } else {
        try {
          errorReported.acquire();
        } catch (InterruptedException e) {
          e.printStackTrace();
          fail("parsing thread interrupted");
        }
      }
    }
  }

  protected abstract String getPathPrefix();

  /** Process interfering with parsing of build files.
   *  It waits until parsing of some BUILD file is started and then reports
   *  arbitrary error. It signals that error was submitted so the parsing can be
   *  finished at the end.
   */
  protected class ErrorReporter implements Runnable {
    private final EventHandler eventHandler;
    private final Semaphore parsingStarted;
    private final Semaphore errorReported;

    public ErrorReporter(EventHandler eventHandler, Semaphore first, Semaphore second) {
      this.eventHandler = eventHandler;
      parsingStarted = first;
      errorReported = second;
    }

    @Override
    public void run() {
      try {
        parsingStarted.acquire();
        eventHandler.handle(
            Event.error(Location.fromFile("dummy"), "Error from other " + "thread"));
        errorReported.release();
      } catch (InterruptedException e) {
        e.printStackTrace();
        fail("ErrorReporter thread interrupted");
      }
    }
  }

  /** {@PackageValidator} whose functionality can be swapped out on demand via {@link #setImpl}. */
  protected static class DummyPackageValidator implements PackageValidator {
    private PackageValidator underlying = PackageValidator.NOOP_VALIDATOR;

    /** Sets {@link PackageValidator} implementation to use. */
    public void setImpl(PackageValidator impl) {
      this.underlying = impl;
    }

    @Override
    public void validate(Package pkg) throws InvalidPackageException {
      underlying.validate(pkg);
    }
  }
}
