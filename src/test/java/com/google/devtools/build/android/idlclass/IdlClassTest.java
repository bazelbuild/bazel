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
package com.google.devtools.build.android.idlclass;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Sets;
import com.google.devtools.build.buildjar.proto.JavaCompilation.CompilationUnit;
import com.google.devtools.build.buildjar.proto.JavaCompilation.Manifest;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

/** {@link IdlClass} test */
@RunWith(JUnit4.class)
public class IdlClassTest {

  static final Manifest MANIFEST =
      Manifest.newBuilder()
          .addCompilationUnit(
              CompilationUnit.newBuilder()
                  .setPath("c/g/Foo.java")
                  .setPkg("c.g")
                  .addTopLevel("Foo")
                  .addTopLevel("Bar"))
          .addCompilationUnit(
              CompilationUnit.newBuilder()
                  .setPath("c/g/Bar.java")
                  .setPkg("c.g")
                  .addTopLevel("Bar")
                  .addTopLevel("Bar2"))
          .addCompilationUnit(
              CompilationUnit.newBuilder()
                  // default package
                  .setPath("wrong/source/dir/Baz.java")
                  .addTopLevel("Baz"))
          .build();

  @Rule public TemporaryFolder tempFolder = new TemporaryFolder();

  @Test
  public void generatedPrefixes() {
    Set<Path> idlSources =
        Sets.newHashSet(Paths.get("c/g/Bar.java"), Paths.get("wrong/source/dir/Baz.java"));
    assertThat(IdlClass.getIdlPrefixes(MANIFEST, idlSources))
        .containsExactly("c/g/Bar", "c/g/Bar2", "Baz");
  }

  @Test
  public void idlClass() throws IOException {
    File classJar = tempFolder.newFile("lib.jar");
    File manifestProto = tempFolder.newFile("lib.manifest");
    File tempDir = tempFolder.newFolder("temp_files");
    File outputClassJar = tempFolder.newFile("lib-idl.jar");
    File outputSourceJar = tempFolder.newFile("lib-idl-src.jar");

    List<String> classes =
        Arrays.asList(
            "Baz.class",
            "Baz$0.class",
            "Baz$1.class",
            "c/g/Foo.class",
            "c/g/Foo$0.class",
            "c/g/Foo$Inner.class",
            "c/g/Foo$Inner$InnerMost.class",
            "c/g/Bar.class",
            "c/g/Bar2.class",
            "c/g/Bar$Inner.class",
            "c/g/Bar2$Inner.class");

    try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(classJar.toPath()))) {
      for (String path : classes) {
        zos.putNextEntry(new ZipEntry(path));
      }
    }

    tempFolder.newFolder("c");
    tempFolder.newFolder("c", "g");
    tempFolder.newFolder("wrong");
    tempFolder.newFolder("wrong", "source");
    tempFolder.newFolder("wrong", "source", "dir");
    for (String file : Arrays.asList("c/g/Foo.java", "c/g/Bar.java", "wrong/source/dir/Baz.java")) {
      tempFolder.newFile(file);
    }

    try (OutputStream os = Files.newOutputStream(manifestProto.toPath())) {
      MANIFEST.writeTo(os);
    }

    IdlClass.main(
        new String[]{
            "--manifest_proto",
            manifestProto.toString(),
            "--class_jar",
            classJar.toString(),
            "--output_class_jar",
            outputClassJar.toString(),
            "--output_source_jar",
            outputSourceJar.toString(),
            "--temp_dir",
            tempDir.toString(),
            "--idl_source_base_dir",
            tempFolder.getRoot().getPath(),
            "c/g/Bar.java",
            "wrong/source/dir/Baz.java"
        });

    List<String> classJarEntries = getJarEntries(outputClassJar);
    assertThat(classJarEntries)
        .containsExactly(
            "c/g/Bar.class",
            "c/g/Bar$Inner.class",
            "c/g/Bar2.class",
            "c/g/Bar2$Inner.class",
            "Baz.class",
            "Baz$0.class",
            "Baz$1.class");

    List<String> sourceJarEntries = getJarEntries(outputSourceJar);
    assertThat(sourceJarEntries)
        .containsExactly(
            "c/g/Bar.java",
            "Baz.java");
  }

  private List<String> getJarEntries(File outputJar) throws IOException {
    List<String> jarEntries = new ArrayList<>();
    try (ZipFile zf = new ZipFile(outputJar)) {
      Enumeration<? extends ZipEntry> entries = zf.entries();
      while (entries.hasMoreElements()) {
        String name = entries.nextElement().getName();
        if (name.endsWith("/") || name.equals("META-INF/MANIFEST.MF")) {
          continue;
        }
        jarEntries.add(name);
      }
    }
    return jarEntries;
  }
}
