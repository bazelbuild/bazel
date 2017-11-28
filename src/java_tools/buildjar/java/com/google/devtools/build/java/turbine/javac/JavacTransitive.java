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

import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin;
import com.google.turbine.bytecode.ClassFile;
import com.google.turbine.bytecode.ClassReader;
import com.google.turbine.bytecode.ClassWriter;
import com.google.turbine.deps.Transitive;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.TreeScanner;
import java.io.IOError;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;

/**
 * Collects the minimal compile-time API for symbols in the supertype closure of compiled classes.
 * This allows non-javac header compilations to be performed against a classpath containing only
 * direct dependencies and no transitive dependencies.
 *
 * <p>See {@link Transitive} for the parallel implementation in non-javac turbine.
 */
public class JavacTransitive {

  private final ImmutableSet<Path> platformJars;

  public JavacTransitive(ImmutableSet<Path> platformJars) {
    this.platformJars = platformJars;
  }

  private final Set<ClassSymbol> closure = new LinkedHashSet<>();
  private Map<String, byte[]> transitive = new LinkedHashMap<>();

  /**
   * Collects re-packaged transitive dependencies, and reset shared state. (The instance may be
   * re-used, e.g. after fall back from the reduced classpath optimization.)
   */
  public Map<String, byte[]> collectTransitiveDependencies() {
    Map<String, byte[]> result = transitive;
    transitive = new LinkedHashMap<>();
    return result;
  }

  /**
   * Records the super type closure of all class declarations in the given compilation unit. Called
   * after attribute is complete for a compilation unit.
   */
  public void postAttribute(Env<AttrContext> result) {
    result.toplevel.accept(
        new TreeScanner() {
          @Override
          public void visitClassDef(JCClassDecl tree) {
            recordSuperClosure(tree.sym);
            super.visitClassDef(tree);
          }
        });
  }

  /**
   * Finish collecting and repackaging. Called while compilation state is still available (e.g. file
   * objects are still open). This method should be idempotent, as {@link JavaCompiler}s sometimes
   * get closed twice.
   */
  public void finish() {
    Set<ClassSymbol> directChildren = new LinkedHashSet<>();
    for (ClassSymbol sym : closure) {
      for (Symbol member : sym.getEnclosedElements()) {
        if (member instanceof ClassSymbol) {
          directChildren.add((ClassSymbol) member);
        }
      }
    }
    closure.addAll(directChildren);
    for (ClassSymbol sym : closure) {
      String name = sym.flatName().toString().replace('.', '/');
      if (transitive.containsKey(name)) {
        continue;
      }
      if (StrictJavaDepsPlugin.getJarPath(sym, platformJars) == null) {
        // Don't repackage symbols we wouldn't report in jdeps, e.g. because they're on the
        // bootclasspath.
        continue;
      }
      JavaFileObject jfo = sym.classfile;
      if (jfo == null || jfo.getKind() != Kind.CLASS) {
        continue;
      }
      ClassFile cf;
      try {
        cf = ClassReader.read(ByteStreams.toByteArray(jfo.openInputStream()));
      } catch (IOException e) {
        throw new IOError(e);
      }
      transitive.put(name, ClassWriter.writeClass(Transitive.trimClass(cf)));
    }
    closure.clear();
  }

  private void recordSuperClosure(Symbol bound) {
    if (!(bound instanceof ClassSymbol)) {
      return;
    }
    ClassSymbol info = (ClassSymbol) bound;
    closure.add(info);
    recordSuperClosure(info.getSuperclass().asElement());
    for (Type i : info.getInterfaces()) {
      recordSuperClosure(i.asElement());
    }
  }
}
