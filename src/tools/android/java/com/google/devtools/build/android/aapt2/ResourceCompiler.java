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

package com.google.devtools.build.android.aapt2;

import com.android.builder.core.VariantType;
import com.android.repository.Revision;
import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.devtools.build.android.AaptCommandBuilder;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.logging.Logger;

/** Invokes aapt2 to compile resources. */
public class ResourceCompiler {
  private static final Logger logger = Logger.getLogger(ResourceCompiler.class.getName());

  private final CompilingVisitor compilingVisitor;

  private static class CompileTask implements Callable<Path> {

    private final Path file;
    private final Path compiledResourcesOut;
    private final Path aapt2;
    private final Revision buildToolsVersion;

    public CompileTask(
        Path file, Path compiledResourcesOut, Path aapt2, Revision buildToolsVersion) {
      this.file = file;
      this.compiledResourcesOut = compiledResourcesOut;
      this.aapt2 = aapt2;
      this.buildToolsVersion = buildToolsVersion;
    }

    @Override
    public Path call() throws Exception {
        logger.fine(
            new AaptCommandBuilder(aapt2)
                .forBuildToolsVersion(buildToolsVersion)
                .forVariantType(VariantType.LIBRARY)
                .add("compile")
                .add("-v")
                .add("--legacy")
                .add("-o", compiledResourcesOut.toString())
                .add(file.toString())
                .execute("Compiling " + file));


      String type = file.getParent().getFileName().toString();
      String filename = file.getFileName().toString();
      if (type.startsWith("values")) {
        filename =
            (filename.indexOf('.') != -1 ? filename.substring(0, filename.indexOf('.')) : filename)
                + ".arsc";
      }

      final Path compiledResourcePath =
          compiledResourcesOut.resolve(type + "_" + filename + ".flat");
      Preconditions.checkArgument(
          Files.exists(compiledResourcePath),
          "%s does not exists after aapt2 ran.",
          compiledResourcePath);
      return compiledResourcePath;
    }

    @Override
    public String toString() {
      return "ResourceCompiler.CompileTask(" + file + ")";
    }
  }

  private static class CompilingVisitor extends SimpleFileVisitor<Path> {

    private final ListeningExecutorService executorService;
    private final Path compiledResources;
    private final List<ListenableFuture<Path>> tasks = new ArrayList<>();
    private final Path aapt2;
    private final Revision buildToolsVersion;

    public CompilingVisitor(
        ListeningExecutorService executorService,
        Path compiledResources,
        Path aapt2,
        Revision buildToolsVersion) {
      this.executorService = executorService;
      this.compiledResources = compiledResources;
      this.aapt2 = aapt2;
      this.buildToolsVersion = buildToolsVersion;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
      // Ignore directories and "hidden" files that start with .
      if (!Files.isDirectory(file) && !file.getFileName().toString().startsWith(".")) {
        tasks.add(
            executorService.submit(
                new CompileTask(
                    file,
                    // Creates a relative output path based on the input path under the
                    // compiledResources path.
                    Files.createDirectories(
                        compiledResources.resolve(
                            (file.isAbsolute() ? file.getRoot().relativize(file) : file)
                                .getParent()
                                .getParent())),
                    aapt2,
                    buildToolsVersion)));
      }
      return super.visitFile(file, attrs);
    }

    List<Path> getCompiledArtifacts() throws InterruptedException, ExecutionException {
      return Futures.allAsList(tasks).get();
    }
  }

  /** Creates a new {@link ResourceCompiler}. */
  public static ResourceCompiler create(
      ListeningExecutorService executorService,
      Path compiledResources,
      Path aapt2,
      Revision buildToolsVersion) {
    return new ResourceCompiler(
        new CompilingVisitor(executorService, compiledResources, aapt2, buildToolsVersion));
  }

  private ResourceCompiler(CompilingVisitor compilingVisitor) {
    this.compilingVisitor = compilingVisitor;
  }

  /** Adds a task to compile the directory using aapt2. */
  public void queueDirectoryForCompilation(Path resource) throws IOException {
    Files.walkFileTree(resource, compilingVisitor);
  }

  /** Returns all paths of the aapt2 compiled resources. */
  public List<Path> getCompiledArtifacts() throws InterruptedException, ExecutionException {
    return compilingVisitor.getCompiledArtifacts();
  }
}
