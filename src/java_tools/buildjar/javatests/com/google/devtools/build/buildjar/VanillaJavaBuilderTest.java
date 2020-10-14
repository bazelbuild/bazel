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

package com.google.devtools.build.buildjar;

import static com.google.common.base.StandardSystemProperty.JAVA_HOME;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.buildjar.VanillaJavaBuilder.VanillaJavaBuilderResult;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarOutputStream;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link VanillaJavaBuilder}Test */
@RunWith(JUnit4.class)
public class VanillaJavaBuilderTest {
  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  VanillaJavaBuilderResult run(List<String> args) throws Exception {
    try (VanillaJavaBuilder builder = new VanillaJavaBuilder()) {
      return builder.run(args);
    }
  }

  ImmutableMap<String, byte[]> readJar(File file) throws IOException {
    ImmutableMap.Builder<String, byte[]> result = ImmutableMap.builder();
    try (JarFile jf = new JarFile(file)) {
      Enumeration<JarEntry> entries = jf.entries();
      while (entries.hasMoreElements()) {
        JarEntry je = entries.nextElement();
        result.put(je.getName(), ByteStreams.toByteArray(jf.getInputStream(je)));
      }
    }
    return result.build();
  }

  @Test
  public void hello() throws Exception {
    Path source = temporaryFolder.newFile("Test.java").toPath();
    Path output = temporaryFolder.newFile("out.jar").toPath();
    Files.write(
        source,
        ImmutableList.of(
            "class A {", //
            "}"),
        UTF_8);
    Path sourceJar = temporaryFolder.newFile("src.srcjar").toPath();
    try (OutputStream os = Files.newOutputStream(sourceJar);
        JarOutputStream jos = new JarOutputStream(os)) {
      jos.putNextEntry(new JarEntry("B.java"));
      jos.write("class B {}".getBytes(UTF_8));
    }

    VanillaJavaBuilderResult result =
        run(
            ImmutableList.of(
                "--javacopts",
                "-Xep:FallThrough:ERROR",
                "--",
                "--sources",
                source.toString(),
                "--source_jars",
                sourceJar.toString(),
                "--output",
                output.toString(),
                "--bootclasspath",
                Paths.get(JAVA_HOME.value()).resolve("lib/rt.jar").toString()));

    assertThat(result.output()).isEmpty();
    assertThat(result.ok()).isTrue();

    ImmutableMap<String, byte[]> outputEntries = readJar(output.toFile());
    assertThat(outputEntries.keySet())
        .containsExactly("META-INF/", "META-INF/MANIFEST.MF", "A.class", "B.class");
  }

  @Test
  public void error() throws Exception {
    Path source = temporaryFolder.newFile("Test.java").toPath();
    Path output = temporaryFolder.newFolder().toPath().resolve("out.jar");
    Files.write(
        source,
        ImmutableList.of(
            "class A {", //
            "  void f(int x) {",
            "    switch (x) {",
            "      case 0:",
            "        System.err.println(0);",
            "      case 1:",
            "        System.err.println(0);",
            "    }",
            "  }",
            "}"),
        UTF_8);

    VanillaJavaBuilderResult result =
        run(
            ImmutableList.of(
                "--javacopts",
                "-Xlint:all",
                "-Werror",
                "--",
                "--sources",
                source.toString(),
                "--output",
                output.toString(),
                "--bootclasspath",
                Paths.get(System.getProperty("java.home")).resolve("lib/rt.jar").toString()));

    assertThat(result.output()).contains("possible fall-through");
    assertThat(result.ok()).isFalse();
    assertThat(Files.exists(output)).isFalse();
  }

  @Test
  public void diagnosticWithoutSource() throws Exception {
    Path source = temporaryFolder.newFile("Test.java").toPath();
    Path output = temporaryFolder.newFolder().toPath().resolve("out.jar");
    Files.write(
        source,
        ImmutableList.of(
            "import java.util.ArrayList;",
            "import java.util.List;",
            "abstract class A {",
            "  abstract void f(List<String> xs);",
            "  {",
            "    f(new ArrayList<>());",
            "  }",
            "}"),
        UTF_8);

    VanillaJavaBuilderResult result =
        run(
            ImmutableList.of(
                "--javacopts",
                "-source",
                "7",
                "-Xlint:none",
                "--",
                "--sources",
                source.toString(),
                "--output",
                output.toString()));

    assertThat(result.output()).contains("note: Some messages have been simplified");
    assertThat(result.ok()).isFalse();
    assertThat(Files.exists(output)).isFalse();
  }

  @Test
  public void cleanOutputDirectories() throws Exception {
    Path source = temporaryFolder.newFile("Test.java").toPath();
    Path output = temporaryFolder.newFile("out.jar").toPath();
    Files.write(
        source,
        ImmutableList.of(
            "class A {", //
            "}"),
        UTF_8);
    Path sourceJar = temporaryFolder.newFile("src.srcjar").toPath();
    try (OutputStream os = Files.newOutputStream(sourceJar);
        JarOutputStream jos = new JarOutputStream(os)) {
      jos.putNextEntry(new JarEntry("B.java"));
      jos.write("class B {}".getBytes(UTF_8));
    }

    Path classDir = temporaryFolder.newFolder().toPath();
    Files.write(
        classDir.resolve("extra.class"),
        new byte[] {(byte) 0xca, (byte) 0xfe, (byte) 0xba, (byte) 0xbe});

    VanillaJavaBuilderResult result =
        run(
            ImmutableList.of(
                "--javacopts",
                "-Xep:FallThrough:ERROR",
                "--",
                "--sources",
                source.toString(),
                "--source_jars",
                sourceJar.toString(),
                "--output",
                output.toString(),
                "--bootclasspath",
                Paths.get(System.getProperty("java.home")).resolve("lib/rt.jar").toString()));

    assertThat(result.output()).isEmpty();
    assertThat(result.ok()).isTrue();

    ImmutableMap<String, byte[]> outputEntries = readJar(output.toFile());
    assertThat(outputEntries.keySet())
        .containsExactly("META-INF/", "META-INF/MANIFEST.MF", "A.class", "B.class");
  }

  // suppress unpopular deferred diagnostic notes for sunapi, deprecation, and unchecked
  @Test
  public void testDeferredDiagnostics() throws Exception {
    Path b = temporaryFolder.newFile("B.java").toPath();
    Path a = temporaryFolder.newFile("A.java").toPath();
    Path output = temporaryFolder.newFile("out.jar").toPath();
    Files.write(
        b,
        ImmutableList.of(
            "@Deprecated", //
            "class B {}"),
        UTF_8);
    Files.write(
        a,
        ImmutableList.of(
            "import java.util.*;", //
            "public class A {",
            "  sun.misc.Unsafe theUnsafe;",
            "  B b;",
            "  List l = new ArrayList<>();",
            "}"),
        UTF_8);

    VanillaJavaBuilderResult result =
        run(
            ImmutableList.of(
                "--sources",
                a.toString(),
                b.toString(),
                "--output",
                output.toString(),
                "--bootclasspath",
                Paths.get(System.getProperty("java.home")).resolve("lib/rt.jar").toString()));

    assertThat(result.output()).isEmpty();
    assertThat(result.ok()).isTrue();
  }

  @Test
  public void nativeHeaders() throws Exception {
    Path foo = temporaryFolder.newFile("FooWithNativeMethod.java").toPath();
    Path bar = temporaryFolder.newFile("BarWithNativeMethod.java").toPath();
    Path output = temporaryFolder.newFile("out.jar").toPath();
    Path nativeHeaderOutput = temporaryFolder.newFile("out-native-headers.jar").toPath();
    Files.write(
        foo,
        ImmutableList.of(
            "package test;",
            "public class FooWithNativeMethod {",
            "  public static native byte[] g(String s);",
            "}"),
        UTF_8);
    Files.write(
        bar,
        ImmutableList.of(
            "package test;",
            "public class BarWithNativeMethod {",
            "  public static native byte[] g(String s);",
            "}"),
        UTF_8);

    VanillaJavaBuilderResult result =
        run(
            ImmutableList.of(
                "--javacopts",
                "-Xep:FallThrough:ERROR",
                "--",
                "--sources",
                foo.toString(),
                bar.toString(),
                "--output",
                output.toString(),
                "--native_header_output",
                nativeHeaderOutput.toString(),
                "--bootclasspath",
                Paths.get(System.getProperty("java.home")).resolve("lib/rt.jar").toString()));

    assertThat(result.output()).isEmpty();
    assertThat(result.ok()).isTrue();

    ImmutableMap<String, byte[]> outputEntries = readJar(nativeHeaderOutput.toFile());
    assertThat(outputEntries.keySet())
        .containsAtLeast("test_BarWithNativeMethod.h", "test_FooWithNativeMethod.h");
  }
}
