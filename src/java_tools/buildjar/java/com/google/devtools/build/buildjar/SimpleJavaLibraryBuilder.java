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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.devtools.build.buildjar.javac.JavacRunner;
import com.sun.tools.javac.main.Main.Result;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/** An implementation of the JavaBuilder that uses in-process javac to compile java files. */
public class SimpleJavaLibraryBuilder extends AbstractJavaBuilder {

  @Override
  Result compileSources(JavaLibraryBuildRequest build, JavacRunner javacRunner, PrintWriter err)
      throws IOException {
    String[] javacArguments = makeJavacArguments(build, build.getClassPath());
    return javacRunner.invokeJavac(build.getPlugins(), javacArguments, err);
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

  /**
   * For the build configuration 'build', construct a command line that can be used for a javac
   * invocation.
   */
  protected String[] makeJavacArguments(JavaLibraryBuildRequest build) {
    return makeJavacArguments(build, build.getClassPath());
  }

  /**
   * For the build configuration 'build', construct a command line that can be used for a javac
   * invocation.
   */
  protected String[] makeJavacArguments(JavaLibraryBuildRequest build, String classPath) {
    List<String> javacArguments = createInitialJavacArgs(build, classPath);

    javacArguments.addAll(getAnnotationProcessingOptions(build));

    for (String option : build.getJavacOpts()) {
      if (option.startsWith("-J")) { // ignore the VM options.
        continue;
      }
      if (option.equals("-processor") || option.equals("-processorpath")) {
        throw new IllegalStateException(
            "Using "
                + option
                + " in javacopts is no longer supported."
                + " Use a java_plugin() rule instead.");
      }
      javacArguments.add(option);
    }

    javacArguments.addAll(build.getSourceFiles());
    return javacArguments.toArray(new String[0]);
  }

  /**
   * Given a JavaLibraryBuildRequest, computes the javac options for the annotation processing
   * requested.
   */
  private List<String> getAnnotationProcessingOptions(JavaLibraryBuildRequest build) {
    List<String> args = new ArrayList<>();

    // Javac treats "-processorpath ''" as setting the processor path to an empty list,
    // whereas omitting the option is treated as not having a processor path (which causes
    // processor path searches to fallback to the class path).
    args.add("-processorpath");
    args.add(build.getProcessorPath().isEmpty() ? "" : build.getProcessorPath());

    if (!build.getProcessors().isEmpty() && !build.getSourceFiles().isEmpty()) {
      // ImmutableSet.copyOf maintains order
      ImmutableSet<String> deduplicatedProcessorNames = ImmutableSet.copyOf(build.getProcessors());
      args.add("-processor");
      args.add(Joiner.on(',').join(deduplicatedProcessorNames));

      // Set javac output directory for generated sources.
      if (build.getSourceGenDir() != null) {
        args.add("-s");
        args.add(build.getSourceGenDir());
      }
    } else {
      // This is necessary because some jars contain discoverable annotation processors that
      // previously didn't run, and they break builds if the "-proc:none" option is not passed to
      // javac.
      args.add("-proc:none");
    }

    return args;
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
