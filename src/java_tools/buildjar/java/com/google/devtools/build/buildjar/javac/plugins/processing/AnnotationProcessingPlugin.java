// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.buildjar.javac.plugins.processing;

import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;

import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCClassDecl;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;

/**
 * A plugin that records information about sources generated during annotation
 * processing.
 */
public class AnnotationProcessingPlugin extends BlazeJavaCompilerPlugin {

  private final HashSet<JCCompilationUnit> toplevels = new HashSet<>();
  private final AnnotationProcessingModule processingModule;

  public AnnotationProcessingPlugin(AnnotationProcessingModule processingModule) {
    this.processingModule = processingModule;
  }

  @Override
  public void postAttribute(Env<AttrContext> env) {
    if (toplevels.add(env.toplevel)) {
      recordSymbols(env.toplevel);
    }
  }

  /**
   * For each top-level type, record the path prefixes of that type's class,
   * and all inner and anonymous classes declared inside that type.
   *
   * <p>e.g. for j.c.g.Foo we record the prefix j/c/g/Foo, which will match
   * j/c/g/Foo.class, j/c/g/Foo$Inner.class, j/c/g/Foo$1.class, etc.
   */
  private void recordSymbols(JCCompilationUnit toplevel) {
    if (toplevel.sourcefile == null) {
      return;
    }
    Path sourcePath = Paths.get(toplevel.sourcefile.toUri());
    if (!processingModule.isGeneratedSource(sourcePath)) {
      return;
    }
    String packageBase = "";
    if (toplevel.getPackageName() != null) {
      packageBase = toplevel.getPackageName().toString().replace('.', '/') + "/";
    }
    for (JCTree decl : toplevel.defs) {
      if (decl instanceof JCClassDecl) {
        String pathPrefix = packageBase + ((JCClassDecl) decl).getSimpleName();
        processingModule.recordPrefix(pathPrefix);
      }
    }
  }
}
