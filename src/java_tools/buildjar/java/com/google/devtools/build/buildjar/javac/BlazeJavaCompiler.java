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

package com.google.devtools.build.buildjar.javac;

import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.CompileStates.CompileState;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.util.Context;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

/**
 * An extended version of the javac compiler, providing support for composable static analyses via a
 * plugin mechanism. BlazeJavaCompiler keeps a list of plugins and calls callback methods in those
 * plugins after certain compiler phases. The plugins perform the actual static analyses.
 */
public class BlazeJavaCompiler extends JavaCompiler {

  /** A list of plugins to run at particular points in the compile */
  private final List<BlazeJavaCompilerPlugin> plugins = new ArrayList<>();

  private BlazeJavaCompiler(Context context, Iterable<BlazeJavaCompilerPlugin> plugins) {
    super(context);

    BlazeJavacStatistics.Builder statisticsBuilder =
        context.get(BlazeJavacStatistics.Builder.class);
    // initialize all plugins
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      plugin.init(context, log, this, statisticsBuilder);
      this.plugins.add(plugin);
    }
  }

  /**
   * Adds an initialization hook to the Context, such that requests for a JavaCompiler (i.e., a
   * lookup for 'compilerKey' of our superclass, JavaCompiler) will actually construct and return
   * BlazeJavaCompiler.
   *
   * <p>This is the preferred way for extending behavior within javac, per the documentation in
   * {@link Context}.
   *
   * <p>Prior to JDK-8038455 additional JavaCompilers were created for annotation processing rounds,
   * but we now expect a single compiler instance per compilation. The factory is still seems to be
   * necessary to avoid context-ordering issues, but we assert that the factory is only called once,
   * and save the output after its call for introspection.
   */
  public static void preRegister(
      final Context context, final Iterable<BlazeJavaCompilerPlugin> plugins) {
    context.put(
        compilerKey,
        new Context.Factory<JavaCompiler>() {
          boolean first = true;

          @Override
          public JavaCompiler make(Context c) {
            if (!first) {
              throw new AssertionError("Expected a single creation of BlazeJavaCompiler.");
            }
            first = false;
            return new BlazeJavaCompiler(c, plugins);
          }
        });
  }

  @Override
  public Env<AttrContext> attribute(Env<AttrContext> env) {
    Env<AttrContext> result = super.attribute(env);
    // don't run plugins if there were compilation errors
    boolean errors = errorCount() > 0;
    // Iterate over all plugins, calling their postAttribute methods
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      if (!errors || plugin.runOnAttributionErrors()) {
        plugin.postAttribute(result);
      }
    }

    return result;
  }

  @Override
  protected void flow(Env<AttrContext> env, Queue<Env<AttrContext>> results) {
    boolean isDone = compileStates.isDone(env, CompileState.FLOW);
    super.flow(env, results);
    if (isDone) {
      return;
    }
    // don't run plugins if there were compilation errors
    boolean errors = errorCount() > 0;
    // Iterate over all plugins, calling their postFlow methods
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      if (!errors || plugin.runOnFlowErrors()) {
        plugin.postFlow(env);
      }
    }
  }

  @Override
  public void close() {
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      plugin.finish();
    }
    plugins.clear();
    super.close();
  }

  /**
   * Testing purposes only. Returns true if the collection of plugins in this instance contains one
   * of the provided type.
   */
  boolean pluginsContain(Class<? extends BlazeJavaCompilerPlugin> klass) {
    for (BlazeJavaCompilerPlugin plugin : plugins) {
      if (klass.isInstance(plugin)) {
        return true;
      }
    }
    return false;
  }
}
