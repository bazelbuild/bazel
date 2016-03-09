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

import com.google.common.base.Preconditions;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A processor to apply additional steps to the compiled java classes. It can be used to add code
 * coverage instrumentation for instance.
 */
public abstract class AbstractPostProcessor {
  private static final Map<String, AbstractPostProcessor> postProcessors = new HashMap<>();

  /**
   * Declares a post processor with a name. This name serves as the command line argument to
   * reference a processor.
   *
   * @param name the command line name of the processor
   * @param postProcessor the post processor object
   */
  public static void addPostProcessor(String name, AbstractPostProcessor postProcessor) {
    postProcessors.put(name, postProcessor);
  }

  private JavaLibraryBuildRequest build = null;

  /**
   * Sets the command line arguments for this processor.
   *
   * @param arguments the list of arguments
   *
   * @throws InvalidCommandLineException when the list of arguments for this processors is
   *         incorrect.
   */
  public abstract void setCommandLineArguments(List<String> arguments)
      throws InvalidCommandLineException;

  /**
   * This initializer is outside of the constructor so the arguments are not passed to the
   * descendants.
   */
  void initialize(JavaLibraryBuildRequest build) {
    this.build = build;
  }

  protected String workingPath(String name) {
    Preconditions.checkNotNull(this.build);
    return name;
  }

  protected boolean shouldCompressJar() {
    return build.compressJar();
  }

  protected String getBuildClassDir() {
    return build.getClassDir();
  }

  /**
   * Main interface method of the post processor.
   */
  public abstract void processRequest() throws IOException;

  /**
   * Create an {@link AbstractPostProcessor} using reflection.
   *
   * @param processorName the name of the processor to instantiate. It should exist in the list of
   *        post processors added with the {@link #addPostProcessor(String, AbstractPostProcessor)}
   *        method.
   * @param arguments the list of arguments that should be passed to the processor during
   *        instantiation.
   * @throws InvalidCommandLineException on error creating the processor
   */
  static AbstractPostProcessor create(String processorName, List<String> arguments)
      throws InvalidCommandLineException {
    AbstractPostProcessor processor = postProcessors.get(processorName);
    if (processor == null) {
      throw new InvalidCommandLineException("No such processor '" + processorName + "'");
    }
    processor.setCommandLineArguments(arguments);
    return processor;
  }

  /**
   * Recursively delete the given file, it is unsafe.
   *
   * @param file the file to recursively remove
   */
  protected static void recursiveRemove(File file) {
    if (file.isDirectory()) {
      for (File f : file.listFiles()) {
        recursiveRemove(f);
      }
    } else if (file.exists()) {
      file.delete();
    }
  }
}
