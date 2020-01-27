// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.CompileStates.CompileState;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.tree.JCTree.JCCompilationUnit;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import java.util.Queue;
import javax.annotation.Nullable;
import javax.tools.JavaFileObject;

/**
 * A {@link JavaCompiler} that drops method bodies and top-level blocks after parsing, and runs
 * Strict Java Deps.
 *
 * <p>The information dropped from the AST improves compilation performance and has no effect
 * on the output of header compilation.
 */
class JavacTurbineJavaCompiler extends JavaCompiler implements AutoCloseable {

  @Nullable private final StrictJavaDepsPlugin strictJavaDeps;
  private final JavacTransitive transitive;

  public JavacTurbineJavaCompiler(
      Context context, @Nullable StrictJavaDepsPlugin strictJavaDeps, JavacTransitive transitive) {
    super(context);
    this.strictJavaDeps = strictJavaDeps;
    if (strictJavaDeps != null) {
      strictJavaDeps.init(
          context, Log.instance(context), this, context.get(BlazeJavacStatistics.Builder.class));
    }
    this.transitive = transitive;
  }

  @Override
  protected JCCompilationUnit parse(JavaFileObject javaFileObject, CharSequence charSequence) {
    JCCompilationUnit result = super.parse(javaFileObject, charSequence);
    TreePruner.prune(context, result);
    return result;
  }

  @Override
  public Env<AttrContext> attribute(Env<AttrContext> env) {
    if (compileStates.isDone(env, CompileState.ATTR)) {
      return env;
    }
    Env<AttrContext> result = super.attribute(env);
    if (strictJavaDeps != null) {
      strictJavaDeps.postAttribute(result);
    }
    transitive.postAttribute(result);
    return result;
  }

  @Override
  protected void flow(Env<AttrContext> env, Queue<Env<AttrContext>> results) {
    // skip FLOW (as if -relax was enabled, except -relax is broken for JDK >= 8)
    if (!compileStates.isDone(env, CompileState.FLOW)) {
      compileStates.put(env, CompileState.FLOW);
    }
    results.add(env);
  }

  @Override
  public void close() {
    if (strictJavaDeps != null) {
      strictJavaDeps.finish();
    }
    transitive.finish();
  }

  /**
   * Override the default {@link JavaCompiler} implementation with {@link JavacTurbineJavaCompiler}
   * for the given compilation context.
   */
  public static void preRegister(
      Context context, @Nullable final StrictJavaDepsPlugin sjd, JavacTransitive transitive) {
    context.put(
        compilerKey,
        new Context.Factory<JavaCompiler>() {
          @Override
          public JavaCompiler make(Context c) {
            return new JavacTurbineJavaCompiler(c, sjd, transitive);
          }
        });
  }
}
