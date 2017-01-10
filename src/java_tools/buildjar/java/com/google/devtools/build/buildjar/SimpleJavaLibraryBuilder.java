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

package com.google.devtools.build.buildjar;

import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.devtools.build.buildjar.javac.JavacRunner;
import com.sun.tools.javac.main.Main.Result;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

/** An implementation of the JavaBuilder that uses in-process javac to compile java files. */
public class SimpleJavaLibraryBuilder extends AbstractJavaBuilder {

  @Override
  Result compileSources(JavaLibraryBuildRequest build, JavacRunner javacRunner, PrintWriter err)
      throws IOException {
    return javacRunner.invokeJavac(
        build.getPlugins(), build.toBlazeJavacArguments(build.getClassPath()), err);
  }

  @Override
  protected void prepareSourceCompilation(JavaLibraryBuildRequest build) throws IOException {
    super.prepareSourceCompilation(build);

    // Create sourceGenDir if necessary.
    if (build.getSourceGenDir() != null) {
      File sourceGenDir = new File(build.getSourceGenDir());
      if (sourceGenDir.exists()) {
        try {
          cleanupOutputDirectory(sourceGenDir);
        } catch (IOException e) {
          throw new IOException("Cannot clean output directory '" + sourceGenDir + "'", e);
        }
      }
      sourceGenDir.mkdirs();
    }
  }

  @Override
  public void buildGensrcJar(JavaLibraryBuildRequest build) throws IOException {
    JarCreator jar = new JarCreator(build.getGeneratedSourcesOutputJar());
    jar.setNormalize(true);
    jar.setCompression(build.compressJar());
    jar.addDirectory(build.getSourceGenDir());
    jar.execute();
  }
}
