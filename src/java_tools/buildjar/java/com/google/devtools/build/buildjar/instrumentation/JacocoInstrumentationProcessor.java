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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.Files.newBufferedReader;
import static java.nio.file.Files.newBufferedWriter;
import static java.nio.file.StandardOpenOption.CREATE_NEW;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.JavaLibraryBuildRequest;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.testing.coverage.BranchCoverageDetail;
import com.google.testing.coverage.BranchDetailAnalyzer;
import com.google.testing.coverage.JacocoLCOVFormatter;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.jacoco.core.analysis.Analyzer;
import org.jacoco.core.analysis.CoverageBuilder;
import org.jacoco.core.analysis.IBundleCoverage;
import org.jacoco.core.data.ExecutionDataStore;
import org.jacoco.core.instr.Instrumenter;
import org.jacoco.core.runtime.OfflineInstrumentationAccessGenerator;
import org.jacoco.report.ISourceFileLocator;

/**
 * Instruments compiled java classes using Jacoco instrumentation library and optionally analyzes
 * them to generate a baseline coverage report.
 */
public final class JacocoInstrumentationProcessor {

  public static JacocoInstrumentationProcessor create(List<String> args)
      throws InvalidCommandLineException {
    // Ignore extra arguments for backwards compatibility (they used to contain filters).
    if (args.size() < 1) {
      throw new InvalidCommandLineException(
          "Number of arguments for Jacoco instrumentation should be 1+ (given "
              + args.size()
              + ": pathsForCoverageFile [baselineCoverageFile].");
    }
    Path pathsForCoverageFile = Path.of(args.get(0));
    Path baselineCoverageFile = null;
    if (args.size() > 1) {
      baselineCoverageFile = Path.of(args.get(1));
    }

    return new JacocoInstrumentationProcessor(pathsForCoverageFile, baselineCoverageFile);
  }

  private Path instrumentedClassesDirectory;
  private final Path pathsForCoverageFile;
  @Nullable private final Path baselineCoverageFile;

  private JacocoInstrumentationProcessor(
      Path pathsForCoverageFile, @Nullable Path baselineCoverageFile) {
    this.pathsForCoverageFile = pathsForCoverageFile;
    this.baselineCoverageFile = baselineCoverageFile;
  }

  /**
   * Instruments classes using Jacoco and keeps copies of uninstrumented class files in
   * jacocoMetadataDir, to be zipped up in the output file jacocoMetadataOutput.
   */
  public void processRequest(JavaLibraryBuildRequest build, JarCreator jar) throws IOException {
    // Use a directory for coverage metadata  that is unique to each built jar. Avoids
    // multiple threads performing read/write/delete actions on the instrumented classes directory.
    instrumentedClassesDirectory = getMetadataDirRelativeToJar(build.getOutputJar());
    Files.createDirectories(instrumentedClassesDirectory);
    jar.setNormalize(true);
    jar.setCompression(build.compressJar());
    Instrumenter instr = new Instrumenter(new OfflineInstrumentationAccessGenerator());
    instrumentRecursively(instr, build.getClassDir());
    jar.addDirectory(instrumentedClassesDirectory);
    jar.addEntry(pathsForCoverageFile.toString(), pathsForCoverageFile);
  }

  public void cleanup() throws IOException {
    if (Files.exists(instrumentedClassesDirectory)) {
      MoreFiles.deleteRecursively(
          instrumentedClassesDirectory, RecursiveDeleteOption.ALLOW_INSECURE);
    }
  }

  // Return the path of the coverage metadata directory relative to the output jar path.
  private static Path getMetadataDirRelativeToJar(Path outputJar) {
    return outputJar.resolveSibling(outputJar + "-coverage-metadata");
  }

  /**
   * Runs Jacoco instrumentation processor over all .class files recursively, starting with root.
   */
  private void instrumentRecursively(Instrumenter instr, Path root) throws IOException {
    var emptyExecutionDataStore = new ExecutionDataStore();
    var baselineCoverageBuilder = new CoverageBuilder();
    var baselineCoverageAnalyzer = new Analyzer(emptyExecutionDataStore, baselineCoverageBuilder);
    var baselineBranchDetailAnalyzer = new BranchDetailAnalyzer(emptyExecutionDataStore);

    Files.walkFileTree(
        root,
        new SimpleFileVisitor<>() {
          @Override
          public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
              throws IOException {
            if (!file.getFileName().toString().endsWith(".class")) {
              return FileVisitResult.CONTINUE;
            }
            // TODO(bazel-team): filter with coverage_instrumentation_filter?
            // It's not clear whether there is any advantage in not instrumenting *Test classes,
            // apart from lowering the covered percentage in the aggregate statistics.

            // We first copy the original .class file to our metadata directory, then instrument it
            // and rewrite the instrumented version back into the regular classes output directory.

            // Not moving or unlinking the source .class file is essential to guarantee visiting
            // it only once during recursive directory traversal while also mutating the directory.
            Path instrumentedCopy = file;
            Path absoluteUninstrumentedCopy = Path.of(file + ".uninstrumented");
            Path uninstrumentedCopy =
                instrumentedClassesDirectory.resolve(root.relativize(absoluteUninstrumentedCopy));
            Files.createDirectories(uninstrumentedCopy.getParent());
            Files.copy(file, uninstrumentedCopy);

            byte[] uninstrumentedBytes = Files.readAllBytes(uninstrumentedCopy);
            String location = file.toString();
            try (InputStream input = new ByteArrayInputStream(uninstrumentedBytes);
                OutputStream output =
                    new BufferedOutputStream(
                        Files.newOutputStream(instrumentedCopy, TRUNCATE_EXISTING))) {
              instr.instrument(input, output, location);
            }
            if (baselineCoverageFile != null) {
              baselineCoverageAnalyzer.analyzeClass(uninstrumentedBytes, location);
              baselineBranchDetailAnalyzer.analyzeClass(uninstrumentedBytes, location);
            }

            return FileVisitResult.CONTINUE;
          }
        });

    if (baselineCoverageFile != null) {
      generateBaselineCoverageReport(
          baselineCoverageFile,
          baselineCoverageBuilder.getBundle("isthisevenused"),
          baselineBranchDetailAnalyzer.getBranchDetails());
    }
  }

  private void generateBaselineCoverageReport(
      Path report, IBundleCoverage bundleCoverage, Map<String, BranchCoverageDetail> branchDetails)
      throws IOException {
    ImmutableSet<String> execPathsSet;
    try (var reader = newBufferedReader(pathsForCoverageFile)) {
      execPathsSet = reader.lines().collect(toImmutableSet());
    }

    var formatter = new JacocoLCOVFormatter(execPathsSet);
    try (var writer = new PrintWriter(newBufferedWriter(report, UTF_8, CREATE_NEW))) {
      var visitor = formatter.createVisitor(writer, branchDetails);
      visitor.visitInfo(ImmutableList.of(), ImmutableList.of());
      // Note the API requires a sourceFileLocator because the HTML and XML formatters display a
      // page of code annotated with coverage information. Having the source files is not actually
      // needed for generating the lcov report.
      visitor.visitBundle(
          bundleCoverage,
          new ISourceFileLocator() {
            @Override
            public Reader getSourceFile(String packageName, String fileName) {
              return null;
            }

            @Override
            public int getTabWidth() {
              return 0;
            }
          });
      visitor.visitEnd();
    }
  }
}
