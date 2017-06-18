// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.benchmark.codegenerator;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Main class for generating code. */
public class Main {

  private static final ImmutableSet<String> allowedProjectNames = ImmutableSet.of(
      CodeGenerator.TARGET_A_FEW_FILES,
      CodeGenerator.TARGET_MANY_FILES,
      CodeGenerator.TARGET_LONG_CHAINED_DEPS,
      CodeGenerator.TARGET_PARALLEL_DEPS);

  private static final Logger logger = Logger.getLogger(Main.class.getName());

  public static void main(String[] args) {
    GeneratorOptions opt = null;
    try {
      opt = parseArgs(args);
    } catch (Exception e) {
      if (!e.getMessage().isEmpty()) {
        logger.log(Level.SEVERE, e.getMessage());
      }
      System.exit(1);
    }

    // Generate or modify Java code
    CodeGenerator codeGenerator = new JavaCodeGenerator();
    generateOrModifyProject(opt, codeGenerator);

    // Generate or modify Cpp code
    codeGenerator = new CppCodeGenerator();
    generateOrModifyProject(opt, codeGenerator);
  }

  public static GeneratorOptions parseArgs(String[] args) throws OptionsParsingException {
    GeneratorOptions opt = Options.parse(GeneratorOptions.class, args).getOptions();

    // Check output_dir argument
    if (opt.outputDir.isEmpty()) {
      throw new IllegalArgumentException("--output_dir should not be empty.");
    }
    if (opt.modificationMode) {
      File dir = new File(opt.outputDir);
      if (!(dir.exists() && dir.isDirectory())) {
        throw new IllegalArgumentException(
            "--output_dir (" + opt.outputDir + ") does not contain code for modification.");
      }
    }
    // Check at least one type of package will be generated
    if (opt.projectNames.isEmpty()) {
      System.err.println(Options.getUsage(GeneratorOptions.class));
      throw new IllegalArgumentException("No type of package is specified.");
    }
    for (String projectName : opt.projectNames) {
      if (!allowedProjectNames.contains(projectName)) {
        throw new IllegalArgumentException("Project name " + projectName + " is not allowed.");
      }
    }

    return opt;
  }

  private static void generateOrModifyProject(GeneratorOptions opt, CodeGenerator codeGenerator) {
    if (opt.modificationMode) {
      codeGenerator.modifyExistingProject(
          opt.outputDir + codeGenerator.getDirSuffix(), ImmutableSet.copyOf(opt.projectNames));
    } else {
      codeGenerator.generateNewProject(
          opt.outputDir + codeGenerator.getDirSuffix(), ImmutableSet.copyOf(opt.projectNames));
    }
  }
}
