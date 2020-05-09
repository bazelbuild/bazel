// Copyright 2011 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.buildjar.javac.plugins;

import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.statistics.BlazeJavacStatistics;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;
import java.util.List;

/**
 * An interface for additional static analyses that need access to the javac compiler's AST at
 * specific points in the compilation process. This class provides callbacks after the attribute and
 * flow phases of the javac compilation process. A static analysis may be implemented by subclassing
 * this abstract class and performing the analysis in the callback methods. The analysis may then be
 * registered with the BlazeJavaCompiler to be run during the compilation process. See {@link
 * com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin} for an example.
 */
public abstract class BlazeJavaCompilerPlugin {

  protected Context context;
  protected Log log;
  protected JavaCompiler compiler;
  protected BlazeJavacStatistics.Builder statisticsBuilder;

  /**
   * Preprocess the command-line flags that were passed to javac. This is called before {@link
   * #init(Context, Log, JavaCompiler, BlazeJavacStatistics.Builder)} and {@link
   * #initializeContext(Context)}.
   *
   * @param args The command-line flags that javac was invoked with.
   * @throws InvalidCommandLineException if the arguments are invalid
   * @return The flags that do not belong to this plugin.
   */
  public List<String> processArgs(List<String> args) throws InvalidCommandLineException {
    return args;
  }

  /**
   * Called after all plugins have processed arguments and can be used to customize the Java
   * compiler context.
   */
  public void initializeContext(Context context) {
    this.context = context;
  }

  /**
   * Performs analysis actions after the attribute phase of the javac compiler. The attribute phase
   * performs symbol resolution on the parse tree.
   *
   * @param env The attributed parse tree (after symbol resolution)
   */
  public void postAttribute(Env<AttrContext> env) {}

  /**
   * Performs analysis actions after the flow phase of the javac compiler. The flow phase performs
   * dataflow checks, such as finding unreachable statements.
   *
   * @param env The attributed parse tree (after symbol resolution)
   */
  public void postFlow(Env<AttrContext> env) {}

  /**
   * Performs analysis actions when the compiler is done and is about to wipe clean its internal
   * data structures (such as the symbol table).
   */
  public void finish() {}

  /**
   * Initializes the plugin. Called by {@link
   * com.google.devtools.build.buildjar.javac.BlazeJavaCompiler}'s constructor.
   *
   * @param context The Context object from the enclosing BlazeJavaCompiler instance
   * @param log The Log object from the enclosing BlazeJavaCompiler instance
   * @param compiler The enclosing BlazeJavaCompiler instance
   * @param statisticsBuilder The builder object for statistics, so that this plugin may report
   *     performance or auxiliary information.
   */
  public void init(
      Context context,
      Log log,
      JavaCompiler compiler,
      BlazeJavacStatistics.Builder statisticsBuilder) {
    this.context = context;
    this.log = log;
    this.compiler = compiler;
    this.statisticsBuilder = statisticsBuilder;
  }

  /** Returns true if the plugin should run on compilations with attribution errors. */
  public boolean runOnAttributionErrors() {
    return false;
  }
}
