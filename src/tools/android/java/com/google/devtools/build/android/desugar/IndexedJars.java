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
package com.google.devtools.build.android.desugar;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import javax.annotation.Nullable;

/**
 * Opens the given list of Jar files and compute an index of all classes in them, to avoid
 * scanning all Jars over and over for each class to load. An indexed jars can have a parent
 * that is firstly used when a file name is searched.
 */
class IndexedJars {

  private final Map<String, JarFile> jarfiles = new HashMap<>();

  /**
   * Parent indexed jars to use before to search a file name into this indexed jars.
   */
  @Nullable
  private final IndexedJars parentIndexedJar;

  /**
   * Index a list of Jar files without a parent indexed jars.
   */
  public IndexedJars(List<Path> jarFiles) throws IOException {
    this(jarFiles, null);
  }

  /**
   * Index a list of Jar files and set a parent indexed jars that is firstly used during the search
   * of a file name.
   */
  public IndexedJars(List<Path> jarFiles, @Nullable IndexedJars parentIndexedJar)
      throws IOException {
    this.parentIndexedJar = parentIndexedJar;
    for (Path jarfile : jarFiles) {
      indexJar(jarfile);
    }
  }

  @Nullable
  public JarFile getJarFile(String filename) {
    Preconditions.checkArgument(filename.endsWith(".class"));

    if (parentIndexedJar != null) {
      JarFile jarFile = parentIndexedJar.getJarFile(filename);
      if (jarFile != null) {
        return jarFile;
      }
    }

    return jarfiles.get(filename);
  }

  private void indexJar(Path jarfile) throws IOException {
    JarFile jar = new JarFile(jarfile.toFile());
    for (Enumeration<JarEntry> cur = jar.entries(); cur.hasMoreElements(); ) {
      JarEntry entry = cur.nextElement();
      if (entry.getName().endsWith(".class") && !jarfiles.containsKey(entry.getName())) {
        jarfiles.put(entry.getName(), jar);
      }
    }
  }
}
