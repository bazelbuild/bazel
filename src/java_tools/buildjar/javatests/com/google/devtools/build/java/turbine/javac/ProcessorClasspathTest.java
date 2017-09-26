// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.turbine.javac;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import java.io.IOError;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.TypeElement;
import javax.tools.FileObject;
import javax.tools.StandardLocation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for isolated processor classloading. */
@RunWith(JUnit4.class)
public class ProcessorClasspathTest extends AbstractJavacTurbineCompilationTest {

  @SupportedAnnotationTypes("*")
  public static class HostClasspathProcessor extends AbstractProcessor {

    @Override
    public SourceVersion getSupportedSourceVersion() {
      return SourceVersion.latest();
    }

    boolean first = true;

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
      if (!first) {
        return false;
      }
      first = false;

      String message;
      try {
        JavacTurbine.class.toString();
        message = "ok";
      } catch (Throwable e) {
        StringWriter stringWriter = new StringWriter();
        e.printStackTrace(new PrintWriter(stringWriter));
        message = stringWriter.toString();
      }
      try {
        FileObject fileObject =
            processingEnv
                .getFiler()
                .createResource(StandardLocation.CLASS_OUTPUT, "", "result.txt");
        try (OutputStream os = fileObject.openOutputStream()) {
          os.write(message.getBytes(UTF_8));
        }
      } catch (IOException e) {
        throw new IOError(e);
      }
      return false;
    }
  }

  @Test
  public void maskProcessorClasspath() throws Exception {
    addSourceLines("MyAnnotation.java", "public @interface MyAnnotation {}");
    addSourceLines("Hello.java", "@MyAnnotation class Hello {}");

    // create a jar containing only HostClasspathProcessor
    Path processorJar = createClassJar("libprocessor.jar", HostClasspathProcessor.class);

    optionsBuilder.addProcessors(ImmutableList.of(HostClasspathProcessor.class.getName()));
    optionsBuilder.addProcessorPathEntries(ImmutableList.of(processorJar.toString()));
    optionsBuilder.addClassPathEntries(ImmutableList.<String>of());
    optionsBuilder.addSources(sources.stream().map(Object::toString).collect(toImmutableList()));

    compile();

    Map<String, byte[]> outputs = collectOutputs();
    assertThat(outputs).containsKey("result.txt");

    String text = new String(outputs.get("result.txt"), UTF_8);
    assertThat(text)
        .contains(
            "java.lang.NoClassDefFoundError:"
                + " com/google/devtools/build/java/turbine/javac/JavacTurbine");
  }
}
