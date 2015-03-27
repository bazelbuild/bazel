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
import com.google.common.io.CharStreams;
import com.google.devtools.build.singlejar.ZipCombiner;
import com.google.devtools.build.xcode.zip.ZipInputEntry;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.annotation.Nullable;

/** Utility code for working with {@link Wrapper}s. */
public class Wrappers {
  private Wrappers() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Executes the command specified by argsArray and wrapper, writing the output directly to this
   * Java process's stdout/stderr. Calling this method should be the last thing you do in
   * {@code main}, because it may exit prematurely with {@link System#exit(int)}.
   */
  public static void executePipingOutput(String[] argsArray, Wrapper wrapper)
      throws IOException, InterruptedException {
    try {
      execute(argsArray, wrapper, true);
    } catch (CommandFailedException e) {
      handleException(e);
    }
  }

  /**
   * Runs the given wrapper using command-line arguments passed to the {@code main} method, and
   * returns the stdout and stderr of the process.
   *
   * @throws CommandFailedException if the command did not successfully run
   */
  public static OutErr executeCapturingOutput(String[] argsArray, Wrapper wrapper)
      throws CommandFailedException, IOException, InterruptedException {
    return execute(argsArray, wrapper, false);
  }

  /**
   * Outputs stdout and stderr, then exits with a non-zero status code.
   */
  public static void handleException(CommandFailedException e) {
    e.outErr.print();
    System.err.print(e.getMessage());
    System.exit(1);
  }

  @Nullable
  private static OutErr execute(String[] argsArray, Wrapper wrapper, boolean printOutput)
      throws CommandFailedException, IOException, InterruptedException {
    FileSystem filesystem = FileSystems.getDefault();
    ArgumentsParsing argsParsing = ArgumentsParsing.parse(
        filesystem, argsArray, wrapper.name(), wrapper.subtoolName());
    for (String error : argsParsing.error().asSet()) {
      throw new CommandFailedException(error, OutErr.empty());
    }
    if (!argsParsing.arguments().isPresent()) {
      throw new IllegalStateException("No arguments or error present");
    }
    Arguments args = argsParsing.arguments().get();
    Path tempDir = getTempDir(filesystem);
    Path outputDir = Files.createTempDirectory(tempDir, "ZippingOutput");
    Path rootedOutputDir = outputDir.resolve(args.bundleRoot());
    Files.createDirectories(
        wrapper.outputDirectoryMustExist() ? rootedOutputDir : rootedOutputDir.getParent());

    ImmutableList<String> subCommandArguments =
        ImmutableList.copyOf(wrapper.subCommand(args, rootedOutputDir.toString()));
    ProcessBuilder processBuilder = new ProcessBuilder(subCommandArguments);
    if (printOutput) {
      processBuilder = processBuilder.inheritIO();
    }
    Process subProcess = processBuilder.start();
    int exit = subProcess.waitFor();
    OutErr outErr = new OutErr(
        streamToString(subProcess.getInputStream()),
        streamToString(subProcess.getErrorStream()));
    if (exit != 0) {
      throw new CommandFailedException("", outErr);
    }

    try (OutputStream out = Files.newOutputStream(Paths.get(args.outputZip()));
        ZipCombiner combiner = new ZipCombiner(out)) {
      ZipInputEntry.addAll(combiner, ZipInputEntry.fromDirectory(outputDir));
    }
    return outErr;
  }

  private static String streamToString(InputStream stream) throws IOException {
    return CharStreams.toString(new InputStreamReader(stream, StandardCharsets.UTF_8));
  }

  private static Path getTempDir(FileSystem filesystem) {
    String tempDir = System.getenv("TMPDIR");
    if (tempDir == null) {
      tempDir = "/tmp";
    }
    return filesystem.getPath(tempDir);
  }

  /** Thrown if command exception fails for some reason. */
  public static class CommandFailedException extends Exception {
    private OutErr outErr;

    public CommandFailedException(String message, OutErr outErr) {
      super(message);
      this.outErr = outErr;
    }

    public OutErr outErr() {
      return outErr;
    }
  }

  /** Stdout and stderr of a process. */
  public static class OutErr {
    public final String stdout;
    public final String stderr;

    public OutErr(String stdout, String stderr) {
      this.stdout = stdout;
      this.stderr = stderr;
    }

    public static OutErr empty() {
      return new OutErr("", "");
    }

    public void print() {
      System.out.print(stdout);
      System.err.print(stderr);
    }
  }
}
