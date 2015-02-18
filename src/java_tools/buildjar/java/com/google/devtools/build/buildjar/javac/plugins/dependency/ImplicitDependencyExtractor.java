// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.buildjar.javac.plugins.dependency;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.view.proto.Deps;

import com.sun.nio.zipfs.ZipFileSystem;
import com.sun.nio.zipfs.ZipPath;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.file.ZipArchive;
import com.sun.tools.javac.file.ZipFileIndexArchive;
import com.sun.tools.javac.nio.JavacPathFileManager;
import com.sun.tools.javac.util.Context;

import java.io.File;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

import javax.lang.model.util.SimpleTypeVisitor7;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;

/**
 * A lightweight mechanism for extracting compile-time dependencies from javac, by performing a scan
 * of the symbol table after compilation finishes. It only includes dependencies from jar files,
 * which can be interface jars or regular third_party jars, matching the compilation model of Blaze.
 * Note that JDK8 may provide support for extracting per-class, finer-grained dependencies, and if
 * that implementation has reasonable overhead it may be a future option.
 */
public class ImplicitDependencyExtractor {

  /** Set collecting dependencies names, used for the text output (soon to be removed) */
  private final Set<String> depsSet;
  /** Map collecting dependency information, used for the proto output */
  private final Map<String, Deps.Dependency> depsMap;
  private final TypeVisitor typeVisitor = new TypeVisitor();
  private final JavaFileManager fileManager;

  /**
   * ImplicitDependencyExtractor does not guarantee any ordering of the reported
   * dependencies. Clients should preserve the original classpath ordering
   * if trying to minimize their classpaths using this information.
   */
  public ImplicitDependencyExtractor(Set<String> depsSet, Map<String, Deps.Dependency> depsMap,
      JavaFileManager fileManager) {
    this.depsSet = depsSet;
    this.depsMap = depsMap;
    this.fileManager = unwrapFileManager(fileManager);
  }

  /**
   * Collects the implicit dependencies of the given set of ClassSymbol roots.
   * As we're interested in differentiating between symbols that were just
   * resolved vs. symbols that were fully completed by the compiler, we start
   * the analysis by finding all the implicit dependencies reachable from the
   * given set of roots. For completeness, we then walk the symbol table
   * associated with the given context and collect the jar files of the
   * remaining class symbols found there.
   *
   * @param context compilation context
   * @param roots root classes in the implicit dependency collection
   */
  public void accumulate(Context context, Set<ClassSymbol> roots) {
    Symtab symtab = Symtab.instance(context);
    if (symtab.classes == null) {
      return;
    }

    // Collect transitive references for root types
    for (ClassSymbol root : roots) {
      root.type.accept(typeVisitor, null);
    }

    Set<String> platformJars = getPlatformJars(fileManager);

    // Collect all other partially resolved types
    for (ClassSymbol cs : symtab.classes.values()) {
      if (cs.classfile != null) {
        collectJarOf(cs.classfile, platformJars);
      } else if (cs.sourcefile != null) {
        collectJarOf(cs.sourcefile, platformJars);
      }
    }
  }

  /**
   * Collect the set of jars on the compilation bootclasspath.
   */
  public static Set<String> getPlatformJars(JavaFileManager fileManager) {

    if (fileManager instanceof StandardJavaFileManager) {
      StandardJavaFileManager sjfm = (StandardJavaFileManager) fileManager;
      ImmutableSet.Builder<String> result = ImmutableSet.builder();
      for (File jar : sjfm.getLocation(StandardLocation.PLATFORM_CLASS_PATH)) {
        result.add(jar.toString());
      }
      return result.build();
    }

    if (fileManager instanceof JavacPathFileManager) {
      JavacPathFileManager jpfm = (JavacPathFileManager) fileManager;
      ImmutableSet.Builder<String> result = ImmutableSet.builder();
      for (Path jar : jpfm.getLocation(StandardLocation.PLATFORM_CLASS_PATH)) {
        result.add(jar.toString());
      }
      return result.build();
    }

    // TODO(cushon): Assuming JavacPathFileManager or StandardJavaFileManager is slightly brittle,
    // but in practice those are the only implementations that matter.
    throw new IllegalStateException("Unsupported file manager type: "
        + fileManager.getClass().toString());
  }

  /**
   * Attempts to add the jar associated with the given JavaFileObject, if any,
   * to the collection, filtering out jars on the compilation bootclasspath.
   *
   * @param reference JavaFileObject representing a class or source file
   * @param platformJars classes on javac's bootclasspath
   */
  private void collectJarOf(JavaFileObject reference, Set<String> platformJars) {

    String name = getJarName(fileManager, reference);
    if (name == null) {
      return;
    }

    // Filter out classes in rt.jar
    if (platformJars.contains(name)) {
      return;
    }

    depsSet.add(name);
    if (!depsMap.containsKey(name)) {
      depsMap.put(name, Deps.Dependency.newBuilder()
          .setKind(Deps.Dependency.Kind.IMPLICIT)
          .setPath(name)
          .build());
    }
  }

  public static String getJarName(JavaFileManager fileManager, JavaFileObject file) {
    file = unwrapFileObject(file);

    if (file instanceof ZipArchive.ZipFileObject
        || file instanceof ZipFileIndexArchive.ZipFileIndexFileObject) {
      // getName() will return something like com/foo/libfoo.jar(Bar.class)
      return file.getName().split("\\(")[0];
    }

    if (fileManager instanceof JavacPathFileManager) {
      JavacPathFileManager fm = (JavacPathFileManager) fileManager;
      Path path = fm.getPath(file);
      if (!(path instanceof ZipPath)) {
        return null;
      }
      ZipFileSystem zipfs = ((ZipPath) path).getFileSystem();
      // calls toString() on the path to the zip archive
      return zipfs.toString();
    }

    return null;
  }


  private static class TypeVisitor extends SimpleTypeVisitor7<Void, Void> {
    // TODO(bazel-team): Override the visitor methods we're interested in.
  }

  private static final Class<?> WRAPPED_FILE_OBJECT =
      getClassOrDie("com.sun.tools.javac.api.ClientCodeWrapper$WrappedFileObject");

  private static final java.lang.reflect.Field UNWRAP_FILE_FIELD =
      getFieldOrDie(WRAPPED_FILE_OBJECT, "clientFileObject");

  private static final Class<?> WRAPPED_JAVA_FILE_MANAGER =
      getClassOrDie("com.sun.tools.javac.api.ClientCodeWrapper$WrappedJavaFileManager");

  private static final java.lang.reflect.Field UNWRAP_FILE_MANAGER_FIELD =
      getFieldOrDie(WRAPPED_JAVA_FILE_MANAGER, "clientJavaFileManager");

  private static Class<?> getClassOrDie(String name) {
    try {
      return Class.forName(name);
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage(), e);
    }
  }

  private static java.lang.reflect.Field getFieldOrDie(Class<?> clazz, String name) {
    try {
      java.lang.reflect.Field field = clazz.getDeclaredField(name);
      field.setAccessible(true);
      return field;
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage(), e);
    }
  }

  public static JavaFileObject unwrapFileObject(JavaFileObject file) {
    if (!WRAPPED_FILE_OBJECT.isInstance(file)) {
      return file;
    }
    try {
      return (JavaFileObject) UNWRAP_FILE_FIELD.get(file);
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage());
    }
  }

  public static JavaFileManager unwrapFileManager(JavaFileManager fileManager) {
    if (!WRAPPED_JAVA_FILE_MANAGER.isInstance(fileManager)) {
      return fileManager;
    }
    try {
      return (JavaFileManager) UNWRAP_FILE_MANAGER_FIELD.get(fileManager);
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage());
    }
  }
}
