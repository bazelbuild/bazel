// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.buildjar.javac.plugins.classloader;

import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;
import java.net.URL;
import java.net.URLClassLoader;
import javax.tools.JavaFileManager;

/** A plugin that customizes the Java compiler for the Error Prone plugin. */
public final class ClassLoaderMaskingPlugin extends BlazeJavaCompilerPlugin {

  @Override
  public void initializeContext(Context context) {
    context.put(
        JavaFileManager.class,
        new Context.Factory<JavaFileManager>() {
          @Override
          public JavaFileManager make(Context c) {
            return new JavacFileManager(c, true, null) {
              @Override
              protected ClassLoader getClassLoader(URL[] urls) {
                return new URLClassLoader(urls, makeMaskedClassLoader());
              }
            };
          }
        });
    super.initializeContext(context);
  }

  /**
   * When Bazel invokes JavaBuilder, it puts javac.jar on the bootstrap class path and
   * JavaBuilder_deploy.jar on the user class path. We need Error Prone to be available on the
   * annotation processor path, but we want to mask out any other classes to minimize class version
   * skew.
   */
  private ClassLoader makeMaskedClassLoader() {
    return new ClassLoader(JavacFileManager.class.getClassLoader()) {
      @Override
      protected Class<?> findClass(String name) throws ClassNotFoundException {
        if (name.startsWith("com.google.errorprone.")) {
          return Class.forName(name);
        } else if (name.startsWith("org.checkerframework.dataflow.")) {
          return Class.forName(name);
        } else {
          throw new ClassNotFoundException(name);
        }
      }
    };
  }
}
