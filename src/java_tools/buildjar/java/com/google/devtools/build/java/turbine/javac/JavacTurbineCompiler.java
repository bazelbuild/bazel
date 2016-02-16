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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.buildjar.javac.plugins.dependency.StrictJavaDepsPlugin;
import com.google.devtools.build.java.turbine.javac.ZipOutputFileManager.OutputFileObject;

import com.sun.tools.javac.file.CacheFSInfo;
import com.sun.tools.javac.main.Arguments;
import com.sun.tools.javac.main.CommandLine;
import com.sun.tools.javac.main.JavaCompiler;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Log;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.annotation.Nullable;
import javax.tools.JavaFileManager;
import javax.tools.StandardLocation;

/** Performs a javac-based turbine compilation given a {@link JavacTurbineCompileRequest}. */
public class JavacTurbineCompiler {

  static JavacTurbineCompileResult compile(JavacTurbineCompileRequest request) throws IOException {

    Map<String, OutputFileObject> files = new LinkedHashMap<>();
    boolean success;
    StringWriter sw = new StringWriter();
    Context context = new Context();

    try (PrintWriter pw = new PrintWriter(sw)) {
      ZipOutputFileManager.preRegister(context, files);
      setupContext(context, request.strictJavaDepsPlugin());
      CacheFSInfo.preRegister(context);

      context.put(Log.outKey, pw);

      ZipOutputFileManager fm = (ZipOutputFileManager) context.get(JavaFileManager.class);
      fm.setLocationFromPaths(StandardLocation.SOURCE_PATH, Collections.<Path>emptyList());
      fm.setLocationFromPaths(StandardLocation.CLASS_PATH, request.classPath());
      fm.setLocationFromPaths(StandardLocation.PLATFORM_CLASS_PATH, request.bootClassPath());
      fm.setLocationFromPaths(
          StandardLocation.ANNOTATION_PROCESSOR_PATH, request.processorClassPath());

      String[] javacArgArray = request.javacOptions().toArray(new String[0]);
      javacArgArray = CommandLine.parse(javacArgArray);

      Arguments args = Arguments.instance(context);
      args.init("turbine", javacArgArray);

      fm.setContext(context);
      fm.handleOptions(args.getDeferredFileManagerOptions());

      JavaCompiler comp = JavaCompiler.instance(context);
      if (request.strictJavaDepsPlugin() != null) {
        request.strictJavaDepsPlugin().init(context, Log.instance(context), comp);
      }

      try {
        comp.compile(args.getFileObjects(), args.getClassNames(), null);
        success = comp.errorCount() == 0;
      } catch (Throwable t) {
        t.printStackTrace(pw);
        success = false;
      }
    }

    return new JavacTurbineCompileResult(ImmutableMap.copyOf(files), success, sw, context);
  }

  static void setupContext(Context context, @Nullable StrictJavaDepsPlugin sjd) {
    JavacTurbineJavaCompiler.preRegister(context, sjd);
  }
}
