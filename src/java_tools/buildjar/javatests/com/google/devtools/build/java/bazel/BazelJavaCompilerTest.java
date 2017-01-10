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

package com.google.devtools.build.java.bazel;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.net.URI;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Sanity checks: make sure we can instantiate a working javac compiler. */
@RunWith(JUnit4.class)
public class BazelJavaCompilerTest {

  private static File getTmpDir() {
    String tmpdir = System.getenv("TEST_TMPDIR");
    if (tmpdir == null) {
      // Fall back on the system temporary directory
      tmpdir = System.getProperty("java.io.tmpdir");
    }
    if (tmpdir == null) {
      fail("TEST_TMPDIR environment variable is not set!");
    }
    return new File(tmpdir);
  }

  @Test
  public void testCompilerNewInstance() throws Exception {
    JavaCompiler javac = BazelJavaCompiler.newInstance();

    assertNotNull(javac.getStandardFileManager(null, null, null));

    // This is a simplified pattern of invoking the compiler API. Note, however, that
    // many examples cast to JavacTask or JavacTaskImpl and invoke the phases separately.
    // Currently, either cast will fail with something that looks like classloader issues:
    // "com.sun.tools.javac.api.JavacTask cannot be cast to com.sun.tools.javac.api.JavacTask"
    assertNotNull(javac.getTask(null, null, null, null, null, null));
  }

  @Test
  public void testAllowsJava7LanguageFeatures() throws Exception {
    assertCompileSucceeds(
        "string://Test.java",
        "class Test {"
            + "  void foo(String s) {"
            + "    switch (s) {"
            + "      default:"
            + "        return;"
            + "    }"
            + "  }"
            + "}");
  }

  @Test
  public void testAllowsJava7APIs() throws Exception {
    assertCompileSucceeds("string://Test.java", "import java.nio.file.Files;" + "class Test {}");
  }

  @Test
  public void testJavacOpts() throws Exception {
    // BazelJavaCompiler loads the default opts from JavaBuilder, so testing against
    // the exact options would be brittle. This is a basic sanity check that
    // the default options include the correct -encoding (which is loaded from
    // JavaBuilder), and that BazelJavaCompiler is appending the -bootclasspath.
    List<String> opts = BazelJavaCompiler.getDefaultJavacopts();
    assertTrue(opts.contains("UTF-8"));
    assertTrue(opts.contains("-bootclasspath"));
  }

  private void assertCompileSucceeds(final String uri, final String content) throws Exception {
    JavaCompiler javac = BazelJavaCompiler.newInstance();
    JavaFileObject source =
        new SimpleJavaFileObject(URI.create(uri), JavaFileObject.Kind.SOURCE) {
          @Override
          public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return content;
          }
        };
    StandardJavaFileManager fileManager = javac.getStandardFileManager(null, null, null);
    // setting the output path by passing a flag to getTask is not reliable
    fileManager.setLocation(StandardLocation.CLASS_OUTPUT, Arrays.asList(getTmpDir()));
    DiagnosticCollector<JavaFileObject> messages = new DiagnosticCollector<>();
    JavaCompiler.CompilationTask task =
        javac.getTask(null, fileManager, messages, null, null, Collections.singletonList(source));
    assertTrue(task.call());
    assertTrue(messages.getDiagnostics().isEmpty());
  }
}
