// Copyright 2015 Google Inc. All rights reserved.
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import com.google.devtools.build.java.bazel.BazelJavaCompiler;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import org.objectweb.asm.ClassReader;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Locale;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import javax.tools.ToolProvider;

/**
 * JUnit tests for ijar tool.
 */
@RunWith(JUnit4.class)
public class IjarTests {

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

  DiagnosticCollector<JavaFileObject> diagnostics;

  private JavaCompiler.CompilationTask makeCompilationTask(String... files) throws IOException {
    JavaCompiler compiler = BazelJavaCompiler.newInstance();
    StandardJavaFileManager fileManager = compiler.getStandardFileManager(null, null, null);
    fileManager.setLocation(StandardLocation.CLASS_PATH,
                            Arrays.asList(new File("third_party/ijar/test/interface_ijar_testlib.jar")));
    fileManager.setLocation(StandardLocation.CLASS_OUTPUT,
                            Arrays.asList(getTmpDir()));
    diagnostics = new DiagnosticCollector<JavaFileObject>();
    return compiler.getTask(null,
                            fileManager,
                            diagnostics,
                            Arrays.asList("-Xlint:deprecation"), // used for deprecation tests
                            null,
                            fileManager.getJavaFileObjects(files));
  }

  /**
   * Test that the ijar tool preserves private nested classes as they
   * may be exposed through public API. This test relies on an
   * interface jar provided through the build rule
   * :interface_ijar_testlib and the Java source file
   * PrivateNestedClass.java.
   */
  @Test
  public void testPrivateNestedClass() throws IOException {
    if (!makeCompilationTask("third_party/ijar/test/PrivateNestedClass.java").call()) {
      fail(getFailedCompilationMessage());
    }
  }

  /**
   * Test that the ijar tool preserves annotations, especially @Target
   * meta-annotation.
   */
  @Test
  public void testRestrictedAnnotations() throws IOException {
    assertFalse(makeCompilationTask("third_party/ijar/test/UseRestrictedAnnotation.java").call());
  }

  /**
   * Test that the ijar tool preserves private nested classes as they
   * may be exposed through public API. This test relies on an
   * interface jar provided through the build rule
   * :interface_ijar_testlib and the Java source file
   * PrivateNestedClass.java.
   */
  @Test
  public void testDeprecatedParts() throws IOException {
    if (!makeCompilationTask("third_party/ijar/test/UseDeprecatedParts.java").call()) {
      fail(getFailedCompilationMessage());
    }
    int deprecatedWarningCount = 0;
    for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
      if ((diagnostic.getKind() == Diagnostic.Kind.MANDATORY_WARNING) &&
          // Java 6:
          (diagnostic.getMessage(Locale.ENGLISH).startsWith("[deprecation]") ||
           // Java 7:
           diagnostic.getMessage(Locale.ENGLISH).contains("has been deprecated"))) {
        deprecatedWarningCount++;
      }
    }
    assertEquals(16, deprecatedWarningCount);
  }

  /**
   * Test that the ijar tool preserves EnclosingMethod attributes and doesn't
   * prevent annotation processors from accessing all the elements in a package.
   */
  @Test
  public void testEnclosingMethod() throws IOException {
    JavaCompiler.CompilationTask task = makeCompilationTask("third_party/ijar/test/package-info.java");
    task.setProcessors(Arrays.asList(new AbstractProcessor() {
      @Override
      public Set<String> getSupportedAnnotationTypes() {
        return Collections.singleton("*");
      }

      @Override
      public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        roundEnv.getElementsAnnotatedWith(java.lang.Override.class);
        return true;
      }
    }));
    if (!task.call()) {
      fail(getFailedCompilationMessage());
    }
  }

  @Test
  public void testVerifyStripping() throws Exception {
    ZipFile zip = new ZipFile("third_party/ijar/test/interface_ijar_testlib.jar");
    Enumeration<? extends ZipEntry> entries = zip.entries();
    while (entries.hasMoreElements()) {
      ZipEntry entry = entries.nextElement();
      ClassReader reader = new ClassReader(zip.getInputStream(entry));
      StripVerifyingVisitor verifier = new StripVerifyingVisitor();

      reader.accept(verifier, 0);

      if (verifier.errors.size() > 0) {
        StringBuilder builder = new StringBuilder();
        builder.append("Verification of ");
        builder.append(entry.getName());
        builder.append(" failed: ");
        for (String msg : verifier.errors) {
          builder.append(msg);
          builder.append("\t");
        }
        fail(builder.toString());
      }
    }
  }

  private String getFailedCompilationMessage() {
    StringBuilder builder = new StringBuilder();
    builder.append("Build failed unexpectedly");
    for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
      builder.append(String.format(
          "\t%s line %d column %d: %s",
          diagnostic.getKind().toString(),
          diagnostic.getLineNumber(),
          diagnostic.getColumnNumber(),
          diagnostic.getMessage(Locale.ENGLISH)));
    }
    return builder.toString();
  }
}
