// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.android.idlclass;

import com.beust.jcommander.JCommander;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.android.AndroidOptionsUtils;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.devtools.build.buildjar.proto.JavaCompilation.CompilationUnit;
import com.google.devtools.build.buildjar.proto.JavaCompilation.Manifest;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.List;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * IdlClass post-processes the output of a Java compilation, and produces
 * a jar containing only the class files for sources that were generated
 * from idl processing.
 */
public class IdlClass {

  public static void main(String[] args) throws IOException {
    IdlClassOptions idlClassOptions = new IdlClassOptions();
    String[] preprocessedArgs = AndroidOptionsUtils.runArgFilePreprocessor(args);
    String[] normalizedArgs =
        AndroidOptionsUtils.normalizeBooleanOptions(idlClassOptions, preprocessedArgs);
    JCommander.newBuilder().addObject(idlClassOptions).build().parse(normalizedArgs);
    Preconditions.checkNotNull(idlClassOptions.manifestProto);
    Preconditions.checkNotNull(idlClassOptions.classJar);
    Preconditions.checkNotNull(idlClassOptions.outputClassJar);
    Preconditions.checkNotNull(idlClassOptions.outputSourceJar);
    Preconditions.checkNotNull(idlClassOptions.tempDir);

    List<Path> idlSources = Lists.newArrayList();
    for (String idlSource : idlClassOptions.residue) {
      idlSources.add(Paths.get(idlSource));
    }

    Manifest manifest = readManifest(idlClassOptions.manifestProto);
    writeClassJar(idlClassOptions, idlSources, manifest);
    writeSourceJar(idlClassOptions, idlSources, manifest);
  }

  private static void writeClassJar(IdlClassOptions options,
      List<Path> idlSources, Manifest manifest) throws IOException {
    Path tempDir = options.tempDir.resolve("classjar");
    Set<Path> idlSourceSet = Sets.newLinkedHashSet(idlSources);
    extractIdlClasses(options.classJar, manifest, tempDir, idlSourceSet);
    writeOutputJar(options.outputClassJar, tempDir);
  }

  private static void writeSourceJar(IdlClassOptions options,
      List<Path> idlSources, Manifest manifest) throws IOException {
    Path tempDir = options.tempDir.resolve("sourcejar");
    Path idlSourceBaseDir = options.idlSourceBaseDir;

    for (Path path : idlSources) {
      for (CompilationUnit unit : manifest.getCompilationUnitList()) {
        if (Paths.get(unit.getPath()).equals(path)) {
          String pkg = unit.getPkg();
          Path source = idlSourceBaseDir != null ? idlSourceBaseDir.resolve(path) : path;
          Path target = tempDir.resolve(pkg.replace('.', '/')).resolve(path.getFileName());
          Files.createDirectories(target.getParent());
          Files.copy(source, target);
          break;
        }
      }
    }
    writeOutputJar(options.outputSourceJar, tempDir);
  }

  /**
   * Reads the compilation manifest.
   */
  private static Manifest readManifest(Path path) throws IOException {
    Manifest manifest;
    try (InputStream inputStream = Files.newInputStream(path)) {
      manifest = Manifest.parseFrom(inputStream);
    }
    return manifest;
  }

  /**
   * For each top-level class in the compilation, determine the path prefix of classes corresponding
   * to that compilation unit.
   *
   * <p>Prefixes are used to correctly handle inner classes, e.g. the top-level class "c.g.Foo" may
   * correspond to "c/g/Foo.class" and also "c/g/Foo$Inner.class" or "c/g/Foo$0.class".
   */
  @VisibleForTesting
  static ImmutableSet<String> getIdlPrefixes(Manifest manifest, Set<Path> idlSources) {
    ImmutableSet.Builder<String> prefixes = ImmutableSet.builder();
    for (CompilationUnit unit : manifest.getCompilationUnitList()) {
      if (!idlSources.contains(Paths.get(unit.getPath()))) {
        continue;
      }
      String pkg;
      if (unit.hasPkg()) {
        pkg = unit.getPkg().replace('.', '/') + "/";
      } else {
        pkg = "";
      }
      for (String toplevel : unit.getTopLevelList()) {
        prefixes.add(pkg + toplevel);
      }
    }
    return prefixes.build();
  }

  /**
   * Unzip all the class files that correspond to idl processor- generated sources into the
   * temporary directory.
   */
  private static void extractIdlClasses(
      Path classJar, Manifest manifest, Path tempDir, Set<Path> idlSources) throws IOException {
    ImmutableSet<String> prefixes = getIdlPrefixes(manifest, idlSources);
    try (JarFile jar = new JarFile(classJar.toFile())) {
      Enumeration<JarEntry> entries = jar.entries();
      while (entries.hasMoreElements()) {
        JarEntry entry = entries.nextElement();
        String name = entry.getName();
        if (!name.endsWith(".class")) {
          continue;
        }
        String prefix = name.substring(0, name.length() - ".class".length());
        int idx = prefix.indexOf('$');
        if (idx > 0) {
          prefix = prefix.substring(0, idx);
        }
        if (prefixes.contains(prefix)) {
          Files.createDirectories(tempDir.resolve(name).getParent());
          Files.copy(jar.getInputStream(entry), tempDir.resolve(name));
        }
      }
    }
  }

  /** Writes the generated class files to the output jar. */
  private static void writeOutputJar(Path outputJar, Path tempDir) throws IOException {
    JarCreator output = new JarCreator(outputJar.toString());
    output.setCompression(true);
    output.setNormalize(true);
    output.addDirectory(tempDir.toString());
    output.execute();
  }
}
