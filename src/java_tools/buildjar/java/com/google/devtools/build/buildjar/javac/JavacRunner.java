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

import com.sun.tools.javac.main.Main.Result;

import java.io.PrintWriter;

/**
 * The JavacRunner is a type that can be used to invoke
 * javac and provides a convenient hook for modifications.
 * It is split in two parts: An interface "JavacRunner" and
 * an implementation of that interface, "JavacRunnerImpl".
 *
 * The type is split in two parts to allow us to load
 * the implementation multiple times in different classloaders.
 * This is neccessary, as a single javac can not run multiple
 * times in parallel. By using different classloaders to load
 * different copies of javac in different JavacRunnerImpls,
 * we can run them in parallel.
 *
 * However, since each JavacRunnerImpl will then be loaded
 * in a different classloader, we then would not be able to
 * refer to it by simply declaring a type as "JavacRunnerImpl",
 * as this refers to the JavacRunnerImpl type loaded with the
 * default classloader. Therefore, we'd have to address each
 * of the different JavacRunnerImpls as "Object" and invoke
 * its method via reflection.
 *
 * We can circumvent this problem by declaring an interface
 * that JavacRunnerImpl implements (i.e. JavacRunner).
 * If we always load this super-interface in the default
 * classloader, and make each JavacRunnerImpl (loaded in its
 * own classloader) implement it, we can refer to the
 * JavacRunnerImpls as "JavacRunner"s in the main program.
 * That way, we can avoid using reflection and "Object"
 * to deal with the different JavacRunnerImpls.
 */
public interface JavacRunner {

  Result invokeJavac(String[] args, PrintWriter output);

}
