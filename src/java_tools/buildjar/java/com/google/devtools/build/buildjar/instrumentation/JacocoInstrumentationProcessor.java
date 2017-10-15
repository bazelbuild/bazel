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

import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.JavaLibraryBuildRequest;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import org.jacoco.core.instr.Instrumenter;
import org.jacoco.core.runtime.OfflineInstrumentationAccessGenerator;

/** Instruments compiled java classes using Jacoco instrumentation library. */
public final class JacocoInstrumentationProcessor {

  public static JacocoInstrumentationProcessor create(List<String> args)
      throws InvalidCommandLineException {
    if (args.size() < 2) {
      throw new InvalidCommandLineException(
          "Number of arguments for Jacoco instrumentation should be 2+ (given "
              + args.size()
              + ": metadataOutput metadataDirectory [filters*].");
    }

    // ignoring filters, they weren't used in the previous implementation
    // TODO(bazel-team): filters should be correctly handled
    return new JacocoInstrumentationProcessor(args.get(1), args.get(0));
  }

  private final String metadataDir;
  private final String coverageInformation;
  private final boolean isNewCoverageImplementation;

  private JacocoInstrumentationProcessor(String metadataDir, String coverageInfo) {
    this.metadataDir = metadataDir;
    this.coverageInformation = coverageInfo;
    // This is part of the new Java coverage implementation where JacocoInstrumentationProcessor
    // receives a file that includes the relative paths of the uninstrumented Java files, instead
    // of the metadata jar.
    this.isNewCoverageImplementation = coverageInfo.endsWith(".txt");
  }

  public boolean isNewCoverageImplementation() {
    return isNewCoverageImplementation;
  }

  /**
   * Instruments classes using Jacoco and keeps copies of uninstrumented class files in
   * jacocoMetadataDir, to be zipped up in the output file jacocoMetadataOutput.
   */
  public void processRequest(JavaLibraryBuildRequest build, JarCreator jar) throws IOException {
    // Clean up jacocoMetadataDir to be used by postprocessing steps. This is important when
    // running JavaBuilder locally, to remove stale entries from previous builds.
    if (metadataDir != null) {
      Path workDir = Paths.get(metadataDir);
      if (Files.exists(workDir)) {
        recursiveRemove(workDir);
      }
      Files.createDirectories(workDir);
    }
    if (jar == null) {
      jar = new JarCreator(coverageInformation);
    }
    jar.setNormalize(true);
    jar.setCompression(build.compressJar());
    Instrumenter instr = new Instrumenter(new OfflineInstrumentationAccessGenerator());
    instrumentRecursively(instr, Paths.get(build.getClassDir()));
    jar.addDirectory(metadataDir);
    if (isNewCoverageImplementation) {
      jar.addEntry(coverageInformation, coverageInformation);
    } else {
      jar.execute();
    }
  }

  /**
   * Runs Jacoco instrumentation processor over all .class files recursively, starting with root.
   */
  private void instrumentRecursively(Instrumenter instr, Path root) throws IOException {
    Files.walkFileTree(
        root,
        new SimpleFileVisitor<Path>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            if (!file.getFileName().toString().endsWith(".class")) {
              return FileVisitResult.CONTINUE;
            }
            // TODO(bazel-team): filter with coverage_instrumentation_filter?
            // It's not clear whether there is any advantage in not instrumenting *Test classes,
            // apart from lowering the covered percentage in the aggregate statistics.

            // We first move the original .class file to our metadata directory, then instrument it
            // and output the instrumented version in the regular classes output directory.
            Path instrumentedCopy = file;
            Path uninstrumentedCopy;
            if (isNewCoverageImplementation) {
              Path absoluteUninstrumentedCopy = Paths.get(file + ".uninstrumented");
              uninstrumentedCopy =
                  Paths.get(metadataDir).resolve(root.relativize(absoluteUninstrumentedCopy));
            } else {
              uninstrumentedCopy = Paths.get(metadataDir).resolve(root.relativize(file));
            }
            Files.createDirectories(uninstrumentedCopy.getParent());
            Files.move(file, uninstrumentedCopy);
            try (InputStream input =
                    new BufferedInputStream(Files.newInputStream(uninstrumentedCopy));
                OutputStream output =
                    new BufferedOutputStream(Files.newOutputStream(instrumentedCopy))) {
              instr.instrument(input, output, file.toString());
            }
            return FileVisitResult.CONTINUE;
          }
        });
  }

  // TODO(b/27069912): handle symlinks
  private static void recursiveRemove(Path path) throws IOException {
    Files.walkFileTree(
        path,
        new SimpleFileVisitor<Path>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            Files.delete(file);
            return FileVisitResult.CONTINUE;
          }

          @Override
          public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
            Files.delete(dir);
            return FileVisitResult.CONTINUE;
          }
        });
  }
}
