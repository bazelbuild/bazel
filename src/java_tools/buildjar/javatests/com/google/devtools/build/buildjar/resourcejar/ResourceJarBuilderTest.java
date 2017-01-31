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

package com.google.devtools.build.buildjar.resourcejar;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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

/** {@link com.google.devtools.build.buildjar.resourcejar.ResourceJarBuilder}Test. */
@RunWith(JUnit4.class)
public class ResourceJarBuilderTest {

  @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

  @Test
  public void options() throws IOException {
    ResourceJarOptions options =
        ResourceJarOptionsParser.parse(
            ImmutableList.of(
                "--output",
                "resource.jar",
                "--messages",
                "m1",
                "m2",
                "--resources",
                "r1",
                "r2",
                "--resource_jars",
                "rj1",
                "rj2",
                "--classpath_resources",
                "cr1",
                "cr2"));
    assertThat(options.output()).isEqualTo("resource.jar");
    assertThat(options.messages()).containsExactly("m1", "m2");
    assertThat(options.resources()).containsExactly("r1", "r2");
    assertThat(options.resourceJars()).containsExactly("rj1", "rj2");
    assertThat(options.classpathResources()).containsExactly("cr1", "cr2");
  }

  @Test
  public void resourceJars() throws Exception {
    File output = temporaryFolder.newFile("resources.jar");

    File jar1 = temporaryFolder.newFile("jar1.jar");
    try (JarOutputStream jos = new JarOutputStream(new FileOutputStream(jar1))) {
      jos.putNextEntry(new JarEntry("one/a.properties"));
      jos.putNextEntry(new JarEntry("one/b.properties"));
    }

    File jar2 = temporaryFolder.newFile("jar2.jar");
    try (JarOutputStream jos = new JarOutputStream(new FileOutputStream(jar2))) {
      jos.putNextEntry(new JarEntry("two/c.properties"));
      jos.putNextEntry(new JarEntry("two/d.properties"));
    }

    ResourceJarBuilder.build(
        ResourceJarOptions.builder()
            .setOutput(output.toString())
            .setResourceJars(ImmutableList.of(jar1.toString(), jar2.toString()))
            .build());

    List<String> entries = new ArrayList<>();
    try (JarFile jf = new JarFile(output)) {
      Enumeration<JarEntry> jes = jf.entries();
      while (jes.hasMoreElements()) {
        entries.add(jes.nextElement().getName());
      }
    }

    assertThat(entries)
        .containsExactly(
            "META-INF/",
            "META-INF/MANIFEST.MF",
            "one/",
            "one/a.properties",
            "one/b.properties",
            "two/",
            "two/c.properties",
            "two/d.properties")
        .inOrder();
  }

  @Test
  public void resources() throws Exception {
    File output = temporaryFolder.newFile("resources.jar");

    Path root = temporaryFolder.newFolder().toPath();

    Path r1 = root.resolve("one/a.properties");
    Files.createDirectories(r1.getParent());
    Files.write(r1, "hello".getBytes(UTF_8));

    Path r2 = root.resolve("two/b.properties");
    Files.createDirectories(r2.getParent());
    Files.write(r2, "goodbye".getBytes(UTF_8));

    ResourceJarBuilder.build(
        ResourceJarOptions.builder()
            .setOutput(output.toString())
            .setResources(
                ImmutableList.of(
                    root + ":" + root.relativize(r1), root + ":" + root.relativize(r2)))
            .build());

    List<String> entries = new ArrayList<>();
    try (JarFile jf = new JarFile(output)) {
      Enumeration<JarEntry> jes = jf.entries();
      while (jes.hasMoreElements()) {
        entries.add(jes.nextElement().getName());
      }
    }

    assertThat(entries)
        .containsExactly(
            "META-INF/",
            "META-INF/MANIFEST.MF",
            "one/",
            "one/a.properties",
            "two/",
            "two/b.properties");
  }

  @Test
  public void rootEntries() throws Exception {
    File output = temporaryFolder.newFile("resources.jar");

    Path root = temporaryFolder.newFolder().toPath();

    Path r1 = root.resolve("one/a.properties");
    Files.createDirectories(r1.getParent());
    Files.write(r1, "hello".getBytes(UTF_8));

    Path r2 = root.resolve("two/b.properties");
    Files.createDirectories(r2.getParent());
    Files.write(r2, "goodbye".getBytes(UTF_8));

    ResourceJarBuilder.build(
        ResourceJarOptions.builder()
            .setOutput(output.toString())
            .setClasspathResources(ImmutableList.of(r1.toString(), r2.toString()))
            .build());

    List<String> entries = new ArrayList<>();
    try (JarFile jf = new JarFile(output)) {
      Enumeration<JarEntry> jes = jf.entries();
      while (jes.hasMoreElements()) {
        entries.add(jes.nextElement().getName());
      }
    }

    assertThat(entries)
        .containsExactly("META-INF/", "META-INF/MANIFEST.MF", "a.properties", "b.properties");
  }

  @Test
  public void messages() throws Exception {
    File output = temporaryFolder.newFile("resources.jar");

    Path root = temporaryFolder.newFolder().toPath();

    Path r1 = root.resolve("one/a.xmb");
    Files.createDirectories(r1.getParent());
    Files.write(r1, "hello".getBytes(UTF_8));

    Path r2 = root.resolve("two/b.xmb");
    Files.createDirectories(r2.getParent());
    Files.write(r2, "goodbye".getBytes(UTF_8));

    // empty messages are omitted
    Path r3 = root.resolve("three/c.xmb");
    Files.createDirectories(r3.getParent());
    Files.write(r3, new byte[0]);

    ResourceJarBuilder.build(
        ResourceJarOptions.builder()
            .setOutput(output.toString())
            .setMessages(
                ImmutableList.of(
                    root + ":" + root.relativize(r1),
                    root + ":" + root.relativize(r2),
                    root + ":" + root.relativize(r3)))
            .build());

    List<String> entries = new ArrayList<>();
    try (JarFile jf = new JarFile(output)) {
      Enumeration<JarEntry> jes = jf.entries();
      while (jes.hasMoreElements()) {
        entries.add(jes.nextElement().getName());
      }
    }

    assertThat(entries)
        .containsExactly(
            "META-INF/", //
            "META-INF/MANIFEST.MF",
            "one/",
            "one/a.xmb",
            "two/",
            "two/b.xmb")
        .inOrder();
  }
}
