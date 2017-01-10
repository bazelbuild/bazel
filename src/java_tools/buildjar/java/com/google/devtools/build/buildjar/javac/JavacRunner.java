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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.sun.tools.javac.main.Main.Result;
import java.io.PrintWriter;

/**
 * The JavacRunner is a type that can be used to invoke javac and provides a convenient hook for
 * modifications.
 */
public interface JavacRunner {

  Result invokeJavac(
      ImmutableList<BlazeJavaCompilerPlugin> plugins, String[] args, PrintWriter output);
}
