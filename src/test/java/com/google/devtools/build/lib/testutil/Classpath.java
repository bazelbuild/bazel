// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.util.Preconditions;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.TreeSet;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * A helper class to find all classes on the current classpath. This is used to automatically create
 * JUnit 3 and 4 test suites.
 */
final class Classpath {
  private static final String CLASS_EXTENSION = ".class";

  /**
   * Finds all classes that live in or below the given package.
   */
  static Set<Class<?>> findClasses(String packageName) {
    Set<Class<?>> result = new LinkedHashSet<>();
    String pathPrefix = (packageName + '.').replace('.', '/');
    for (String entryName : getClassPath()) {
      File classPathEntry = new File(entryName);
      if (classPathEntry.exists()) {
        try {
          Set<String> classNames;
          if (classPathEntry.isDirectory()) {
            classNames = findClassesInDirectory(classPathEntry, pathPrefix);
          } else {
            classNames = findClassesInJar(classPathEntry, pathPrefix);
          }
          for (String className : classNames) {
            Class<?> clazz = Class.forName(className);
            result.add(clazz);
          }
        } catch (IOException e) {
          throw new AssertionError("Can't read classpath entry "
              + entryName + ": " + e.getMessage());
        } catch (ClassNotFoundException e) {
          throw new AssertionError("Class not found even though it is on the classpath "
              + entryName + ": " + e.getMessage());
        }
      }
    }
    return result;
  }

  private static Set<String> findClassesInDirectory(File classPathEntry, String pathPrefix) {
    Set<String> result = new TreeSet<>();
    File directory = new File(classPathEntry, pathPrefix);
    innerFindClassesInDirectory(result, directory, pathPrefix);
    return result;
  }

  /**
   * Finds all classes and sub packages in the given directory that are below the given package and
   * add them to the respective sets.
   *
   * @param directory Directory to inspect
   * @param pathPrefix Prefix for the path to the classes that are requested
   *                   (ex: {@code com/google/foo/bar})
   */
  private static void innerFindClassesInDirectory(Set<String> classNames, File directory,
      String pathPrefix) {
    Preconditions.checkArgument(pathPrefix.endsWith("/"));
    if (directory.exists()) {
      for (File f : directory.listFiles()) {
        String name = f.getName();
        if (name.endsWith(CLASS_EXTENSION)) {
          String clzName = getClassName(pathPrefix + name);
          classNames.add(clzName);
        } else if (f.isDirectory()) {
          findClassesInDirectory(f, pathPrefix + name + "/");
        }
      }
    }
  }

  /**
   * Returns a set of all classes in the jar that start with the given prefix.
   */
  private static Set<String> findClassesInJar(File jarFile, String pathPrefix) throws IOException {
    Set<String> classNames = new TreeSet<>();
    try (ZipFile zipFile = new ZipFile(jarFile)) {
      Enumeration<? extends ZipEntry> entries = zipFile.entries();
      while (entries.hasMoreElements()) {
        String entryName = entries.nextElement().getName();
        if (entryName.startsWith(pathPrefix) && entryName.endsWith(CLASS_EXTENSION)) {
          classNames.add(getClassName(entryName));
        }
      }
    }
    return classNames;
  }

  /**
   * Given the absolute path of a class file, return the class name.
   */
  private static String getClassName(String className) {
    int classNameEnd = className.length() - CLASS_EXTENSION.length();
    return className.substring(0, classNameEnd).replace('/', '.');
  }

  private static void getClassPathsFromClasspathJar(File classpathJar, Set<String> classPaths)
      throws IOException {
    Manifest manifest = new JarFile(classpathJar).getManifest();
    Attributes attributes = manifest.getMainAttributes();
    for (String classPath : attributes.getValue("Class-Path").split(" ")) {
      try {
        classPaths.add(Paths.get(new URI(classPath)).toAbsolutePath().toString());
      } catch (URISyntaxException e) {
        throw new AssertionError(
            "Error parsing classpath uri " + classPath + ": " + e.getMessage());
      }
    }
  }

  /**
   * Gets the class path from the System Property "java.class.path" and splits it up into the
   * individual elements.
   *
   * <p>Bazel creates a classpath jar when the class path length exceeds command line length limit,
   * read the class path value from its manifest file if it's a classpath jar.
   */
  private static Set<String> getClassPath() {
    String classPath = System.getProperty("java.class.path");
    String separator = System.getProperty("path.separator", ":");
    String[] classPaths = classPath.split(Pattern.quote(separator));
    Set<String> completeClassPaths = new TreeSet<>();
    for (String entryName : classPaths) {
      completeClassPaths.add(entryName);
      if (entryName.endsWith("-classpath.jar")) {
        File classPathEntry = new File(entryName);
        if (classPathEntry.exists() && classPathEntry.isFile()) {
          try {
            getClassPathsFromClasspathJar(classPathEntry, completeClassPaths);
          } catch (IOException e) {
            throw new AssertionError(
                "Can't read classpath entry " + entryName + ": " + e.getMessage());
          }
        }
      }
    }
    return completeClassPaths;
  }
}
