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

import com.google.common.base.Splitter;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.java.turbine.javac.JavacTurbine.Result;
import com.google.turbine.options.TurbineOptions;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarOutputStream;
import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.util.Textifier;
import org.objectweb.asm.util.TraceClassVisitor;

public abstract class AbstractJavacTurbineCompilationTest {

  @Rule public final TemporaryFolder temp = new TemporaryFolder();

  Path sourcedir;
  List<Path> sources;
  Path tempdir;
  Path output;
  Path outputDeps;

  final TurbineOptions.Builder optionsBuilder = TurbineOptions.builder();

  @Before
  public void setUp() throws IOException {
    sourcedir = temp.newFolder().toPath();
    tempdir = temp.newFolder("_temp").toPath();
    output = temp.newFile("out.jar").toPath();
    outputDeps = temp.newFile("out.jdeps").toPath();

    sources = new ArrayList<>();

    optionsBuilder
        .setOutput(output.toString())
        .setTempDir(tempdir.toString())
        .addBootClassPathEntries(
            Splitter.on(File.pathSeparatorChar)
                .splitToList(System.getProperty("sun.boot.class.path"))
                .stream()
                .map(e -> Paths.get(e).toAbsolutePath().toString())
                .collect(toImmutableList()))
        .setOutputDeps(outputDeps.toString())
        .addAllJavacOpts(Arrays.asList("-source", "8", "-target", "8"))
        .setTargetLabel("//test")
        .setRuleKind("java_library");
  }

  protected void addSourceLines(String path, String... lines) throws IOException {
    Path source = sourcedir.resolve(path);
    sources.add(source);
    Files.write(source, Arrays.asList(lines), UTF_8);
  }

  protected void compile() throws IOException {
    optionsBuilder.addSources(sources.stream().map(p -> p.toString()).collect(toImmutableList()));
    try (JavacTurbine turbine =
        new JavacTurbine(
            new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.err, UTF_8))),
            optionsBuilder.build())) {
      assertThat(turbine.compile()).isEqualTo(Result.OK_WITH_REDUCED_CLASSPATH);
    }
  }

  protected Map<String, byte[]> collectOutputs() throws IOException {
    return collectFiles(output);
  }

  static Map<String, byte[]> collectFiles(Path jar) throws IOException {
    Map<String, byte[]> files = new LinkedHashMap<>();
    try (JarFile jf = new JarFile(jar.toFile())) {
      Enumeration<JarEntry> entries = jf.entries();
      while (entries.hasMoreElements()) {
        JarEntry entry = entries.nextElement();
        files.put(entry.getName(), ByteStreams.toByteArray(jf.getInputStream(entry)));
      }
    }
    return files;
  }

  static String textify(byte[] bytes) {
    StringWriter sw = new StringWriter();
    ClassReader cr = new ClassReader(bytes);
    cr.accept(new TraceClassVisitor(null, new Textifier(), new PrintWriter(sw, true)), 0);
    return sw.toString();
  }

  protected Path createClassJar(String jarName, Class<?>... classes) throws IOException {
    Path jarPath = temp.newFile(jarName).toPath();
    try (OutputStream os = Files.newOutputStream(jarPath);
        JarOutputStream jos = new JarOutputStream(os)) {
      for (Class<?> clazz : classes) {
        String classFileName = clazz.getName().replace('.', '/') + ".class";
        jos.putNextEntry(new JarEntry(classFileName));
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(classFileName)) {
          ByteStreams.copy(is, jos);
        }
      }
    }
    return jarPath;
  }
}
