// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.view.java;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import java.util.jar.JarFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * End-to-end test for java annotation processing functionality. Creates a package with an
 * annotation processor that writes into a file that is packaged into the generated jar. Then tests
 * that the file is present in the jar.
 */
@RunWith(JUnit4.class)
public class JavaAnnotationTest extends BuildIntegrationTestCase {

  @Test
  public void testCustomAnnotationProcessor() throws Exception {
    write(
        "java/test/processor/BUILD",
        "java_library(name = 'annotation',",
        "   srcs = [ 'TestAnnotation.java' ])",
        "java_plugin(name = 'processor',",
        "   srcs = [ 'Processor.java' ],",
        "   deps = [ ':annotation' ],",
        "   processor_class = 'test.processor.Processor' )");

    write(
        "java/test/processor/TestAnnotation.java",
        "package test.processor;",
        "import java.lang.annotation.*;",
        "@Target(value = {ElementType.TYPE})",
        "public @interface TestAnnotation {",
        "}");

    write(
        "java/test/processor/Processor.java",
        "package test.processor;",
        "import java.util.*;",
        "import java.io.*;",
        "import javax.annotation.processing.*;",
        "import javax.tools.*;",
        "import javax.lang.model.*;",
        "import javax.lang.model.element.*;",
        "@SupportedAnnotationTypes(value= {\"test.processor.TestAnnotation\"})",
        "public class Processor extends AbstractProcessor {",
        "  private ProcessingEnvironment mainEnvironment;",
        "  public void init(ProcessingEnvironment environment) {",
        "    mainEnvironment = environment;",
        "  }",
        "  public boolean process(Set<? extends TypeElement> annotations,",
        "      RoundEnvironment roundEnv) {",
        "    Filer filer = mainEnvironment.getFiler();",
        "    try {",
        "      FileObject output = filer.createResource(StandardLocation.CLASS_OUTPUT, ",
        "          \"\", \"testfile\");",
        "      PrintWriter writer = new PrintWriter(output.openWriter());",
        "      writer.write(\"Annotation Processing Done.\");",
        "    } catch (IOException ex) {",
        "      return false;",
        "    }",
        "    return true;",
        "  }",
        "}");

    write(
        "java/test/client/BUILD",
        "java_library(name = 'client',",
        "   srcs = [ 'ProcessorClient.java' ],",
        "   deps = [ '//java/test/processor:annotation' ],",
        "   plugins = [ '//java/test/processor:processor' ] )");

    write(
        "java/test/client/ProcessorClient.java",
        "package test.client;",
        "import test.processor.TestAnnotation;",
        "@TestAnnotation()",
        "class ProcessorClient { }");

    buildTarget("//java/test/client:client");
    Iterable<Artifact> artifacts = getArtifacts("//java/test/client:libclient.jar");
    String path = artifacts.iterator().next().getPath().getPathString();
    try (JarFile jar = new JarFile(path)) {
      assertThat(jar.getJarEntry("testfile")).isNotNull();
    }
  }
}
