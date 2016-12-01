// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.devtools.build.buildjar.instrumentation;

import com.google.common.io.Files;
import com.google.devtools.build.buildjar.AbstractPostProcessor;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import org.jacoco.core.instr.Instrumenter;
import org.jacoco.core.runtime.OfflineInstrumentationAccessGenerator;

/** Instruments compiled java classes using Jacoco instrumentation library. */
public final class JacocoInstrumentationProcessor extends AbstractPostProcessor {

  private String metadataDir;
  private String metadataOutput;

  @Override
  public void setCommandLineArguments(List<String> args) throws InvalidCommandLineException {
    if (args.size() < 2) {
      throw new InvalidCommandLineException(
          "Number of arguments for Jacoco instrumentation should be 2+ (given "
              + args.size()
              + ": metadataOutput metadataDirectory [filters*].");
    }

    metadataDir = args.get(1);
    metadataOutput = args.get(0);
    // ignoring filters, they weren't used in the previous implementation
    // TODO(bazel-team): filters should be correctly handled
  }

  /**
   * Instruments classes using Jacoco and keeps copies of uninstrumented class files in
   * jacocoMetadataDir, to be zipped up in the output file jacocoMetadataOutput.
   */
  @Override
  public void processRequest() throws IOException {
    // Clean up jacocoMetadataDir to be used by postprocessing steps. This is important when
    // running JavaBuilder locally, to remove stale entries from previous builds.
    if (metadataDir != null) {
      File workDir = new File(workingPath(metadataDir));
      if (workDir.exists()) {
        recursiveRemove(workDir);
      }
      workDir.mkdirs();
    }

    JarCreator jar = new JarCreator(workingPath(metadataOutput));
    jar.setNormalize(true);
    jar.setCompression(shouldCompressJar());
    Instrumenter instr = new Instrumenter(new OfflineInstrumentationAccessGenerator());
    // TODO(bazel-team): not sure whether Emma did anything fancier than this (multithreaded?)
    instrumentRecursively(instr, new File(workingPath(getBuildClassDir())));
    jar.addDirectory(workingPath(metadataDir));
    jar.execute();
  }

  /**
   * Runs Jacoco instrumentation processor over all .class files recursively, starting with root.
   */
  private void instrumentRecursively(Instrumenter instr, File root) throws IOException {
    for (File f : Files.fileTreeTraverser().preOrderTraversal(root).filter(Files.isFile())) {
      if (f.isDirectory()) {
        instrumentRecursively(instr, f);
      } else if (f.isFile() && f.getName().endsWith(".class")) {
        // TODO(bazel-team): filter with coverage_instrumentation_filter?
        // It's not clear whether there is any advantage in not instrumenting *Test classes, apart
        // from lowering the covered percentage in the aggregate statistics.

        // We first move the original .class file to our metadata directory, then instrument it and
        // output the instrumented version in the regular classes output directory.
        File instrumentedCopy = new File(f.getPath());
        File uninstrumentedCopy =
            new File(
                workingPath(f.getPath().replace(workingPath(getBuildClassDir()), metadataDir)));
        uninstrumentedCopy.getParentFile().mkdirs();
        f.renameTo(uninstrumentedCopy);

        try (InputStream input = new BufferedInputStream(new FileInputStream(uninstrumentedCopy));
            OutputStream output =
                new BufferedOutputStream(new FileOutputStream(instrumentedCopy))) {
          instr.instrument(input, output, f.getName());
        }
      }
    }
  }
}
