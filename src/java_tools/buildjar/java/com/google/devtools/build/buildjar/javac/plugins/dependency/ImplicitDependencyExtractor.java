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

import com.google.devtools.build.lib.view.proto.Deps;

import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.file.ZipArchive;
import com.sun.tools.javac.file.ZipFileIndexArchive;
import com.sun.tools.javac.util.Context;

import java.io.IOError;
import java.io.IOException;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.lang.model.util.SimpleTypeVisitor7;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
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
    this.fileManager = fileManager;
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

    Set<JavaFileObject> platformClasses = getPlatformClasses(fileManager);

    // Collect all other partially resolved types
    for (ClassSymbol cs : symtab.classes.values()) {
      if (cs.classfile != null) {
        collectJarOf(cs.classfile, platformClasses);
      } else if (cs.sourcefile != null) {
        collectJarOf(cs.sourcefile, platformClasses);
      }
    }
  }

  /**
   * Collect the set of classes on the compilation bootclasspath.
   *
   * <p>TODO(bazel-team): this needs some work. JavaFileManager.list() is slower than
   * StandardJavaFileManager.getLocation() and doesn't get cached. Additionally, tracking all
   * classes in the bootclasspath requires a much bigger set than just tracking a list of jars.
   * However, relying on the context containing a StandardJavaFileManager is brittle (e.g. Lombok
   * wraps the file-manager in a ForwardingJavaFileManager.)
   */
  public static HashSet<JavaFileObject> getPlatformClasses(JavaFileManager fileManager) {
    HashSet<JavaFileObject> result = new HashSet<JavaFileObject>();
    Iterable<JavaFileObject> files;
    try {
      files = fileManager.list(
        StandardLocation.PLATFORM_CLASS_PATH, "", EnumSet.of(JavaFileObject.Kind.CLASS), true);
    } catch (IOException e) {
      throw new IOError(e);
    }
    for (JavaFileObject file : files) {
      result.add(file);
    }
    return result;
  }

  /**
   * Attempts to add the jar associated with the given JavaFileObject, if any,
   * to the collection, filtering out jars on the compilation bootclasspath.
   *
   * @param reference JavaFileObject representing a class or source file
   * @param platformClasses classes on javac's bootclasspath
   */
  private void collectJarOf(JavaFileObject reference, Set<JavaFileObject> platformClasses) {
    reference = unwrapFileObject(reference);
    if (reference instanceof ZipArchive.ZipFileObject ||
        reference instanceof ZipFileIndexArchive.ZipFileIndexFileObject) {
      // getName() will return something like com/foo/libfoo.jar(Bar.class)
      String name = reference.getName().split("\\(")[0];
      // Filter out classes in rt.jar
      if (!platformClasses.contains(reference)) {
        depsSet.add(name);
        if (!depsMap.containsKey(name)) {
          depsMap.put(name, Deps.Dependency.newBuilder()
              .setKind(Deps.Dependency.Kind.IMPLICIT)
              .setPath(name)
              .build());
        }
      }
    }
  }


  private static class TypeVisitor extends SimpleTypeVisitor7<Void, Void> {
    // TODO(bazel-team): Override the visitor methods we're interested in.
  }

  private static final Class<?> WRAPPED_JAVA_FILE_OBJECT =
      getClassOrDie("com.sun.tools.javac.api.ClientCodeWrapper$WrappedJavaFileObject");

  private static final java.lang.reflect.Field UNWRAP_FIELD =
      getFieldOrDie(
          getClassOrDie("com.sun.tools.javac.api.ClientCodeWrapper$WrappedFileObject"),
          "clientFileObject");

  private static Class<?> getClassOrDie(String name) {
    try {
      return Class.forName(name);
    } catch (ClassNotFoundException e) {
      throw new LinkageError(e.getMessage());
    }
  }

  private static java.lang.reflect.Field getFieldOrDie(Class<?> clazz, String name) {
    try {
      java.lang.reflect.Field field = clazz.getDeclaredField(name);
      field.setAccessible(true);
      return field;
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage());
    }
  }

  public static JavaFileObject unwrapFileObject(JavaFileObject file) {
    if (!file.getClass().equals(WRAPPED_JAVA_FILE_OBJECT)) {
      return file;
    }
    try {
      return (JavaFileObject) UNWRAP_FIELD.get(file);
    } catch (ReflectiveOperationException e) {
      throw new LinkageError(e.getMessage());
    }
  }
}
