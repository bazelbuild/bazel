// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.zippingoutput;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.xcode.zip.ZipInputEntry;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Utility code for working with {@link Wrapper}s. */
public class Wrappers {
  private Wrappers() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Runs the given wrapper using command-line arguments passed to the {@code main} method. Calling
   * this method should be the last thing you do in {@code main}, because it may exit prematurely
   * with {@link System#exit(int)}.
   */
  public static void execute(String[] argsArray, Wrapper wrapper)
      throws IOException, InterruptedException {
    FileSystem filesystem = FileSystems.getDefault();
    ArgumentsParsing argsParsing = ArgumentsParsing.parse(
        filesystem, argsArray, wrapper.name(), wrapper.subtoolName());
    for (String error : argsParsing.error().asSet()) {
      System.err.printf(error);
      System.exit(1);
    }
    Path tempDir = getTempDir(filesystem);
    for (Arguments args : argsParsing.arguments().asSet()) {
      Path outputDir = Files.createTempDirectory(tempDir, "ZippingOutput");
      Path rootedOutputDir = outputDir.resolve(args.bundleRoot());
      Files.createDirectories(
          wrapper.outputDirectoryMustExist() ? rootedOutputDir : rootedOutputDir.getParent());

      ImmutableList<String> subCommandArguments =
          ImmutableList.copyOf(wrapper.subCommand(args, rootedOutputDir.toString()));
      Process subProcess = new ProcessBuilder(subCommandArguments).inheritIO().start();
      int exit = subProcess.waitFor();
      if (exit != 0) {
        System.exit(exit);
      }

      try (OutputStream out = Files.newOutputStream(Paths.get(args.outputZip()));
          ZipCombiner combiner = new ZipCombiner(out)) {
        ZipInputEntry.addAll(combiner, ZipInputEntry.fromDirectory(outputDir));
      }
    }
  }

  private static Path getTempDir(FileSystem filesystem) {
    String tempDir = System.getenv("TMPDIR");
    if (tempDir == null) {
      tempDir = "/tmp";
    }
    return filesystem.getPath(tempDir);
  }
}
