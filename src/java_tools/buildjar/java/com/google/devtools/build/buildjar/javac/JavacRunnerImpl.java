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

import com.sun.tools.javac.main.Main.Result;

import java.io.PrintWriter;
import java.util.List;

/**
 * This class wraps a single invocation of Javac. We
 * invoke javac statically but wrap it with a synchronization.
 * This is because the same javac cannot be invoked multiple
 * times in parallel.
 */
public class JavacRunnerImpl implements JavacRunner {

  private final List<BlazeJavaCompilerPlugin> plugins;

  /**
   * Passes extra information to BlazeJavacMain in case strict Java
   * dependencies are enforced.
   */
  public JavacRunnerImpl(List<BlazeJavaCompilerPlugin> plugins) {
    this.plugins = plugins;
  }

  @Override
  public synchronized Result invokeJavac(String[] args, PrintWriter output) {
    return new BlazeJavacMain(output, plugins).compile(args);
  }

}
