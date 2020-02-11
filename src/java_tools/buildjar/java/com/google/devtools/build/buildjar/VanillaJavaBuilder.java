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

package com.google.devtools.build.buildjar;

import static com.google.common.base.MoreObjects.firstNonNull;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Locale.ENGLISH;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.io.MoreFiles;
import com.google.common.io.RecursiveDeleteOption;
import com.google.devtools.build.buildjar.jarhelper.JarCreator;
import com.google.devtools.build.buildjar.javac.JavacOptions;
import com.google.devtools.build.buildjar.proto.JavaCompilation.Manifest;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URI;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.processing.Processor;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaCompiler.CompilationTask;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import javax.tools.ToolProvider;

/**
 * A JavaBuilder that supports non-standard JDKs and unmodified javac's.
 *
 * <p>Does not support:
 *
 * <ul>
 *   <li>Error Prone
 *   <li>strict Java deps
 *   <li>header compilation
 *   <li>Android desugaring
 *   <li>coverage instrumentation
 *   <li>genclass handling for IDEs
 * </ul>
 */
public class VanillaJavaBuilder implements Closeable {

  /** Cache of opened zip filesystems. */
  private final Map<Path, FileSystem> filesystems = new HashMap<>();

  private FileSystem getJarFileSystem(Path sourceJar) throws IOException {
    FileSystem fs = filesystems.get(sourceJar);
    if (fs == null) {
      filesystems.put(sourceJar, fs = FileSystems.newFileSystem(sourceJar, null));
    }
    return fs;
  }

  public static void main(String[] args) throws IOException {
    if (args.length == 1 && args[0].equals("--persistent_worker")) {
      System.exit(runPersistentWorker());
    } else {
      try (VanillaJavaBuilder builder = new VanillaJavaBuilder()) {
        VanillaJavaBuilderResult result = builder.run(ImmutableList.copyOf(args));
        System.err.print(result.output());
        System.exit(result.ok() ? 0 : 1);
      }
    }
  }

  private static int runPersistentWorker() {
    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);
        if (request == null) {
          break;
        }
        try (VanillaJavaBuilder builder = new VanillaJavaBuilder()) {
          VanillaJavaBuilderResult result = builder.run(request.getArgumentsList());
          WorkResponse response =
              WorkResponse.newBuilder()
                  .setOutput(result.output())
                  .setExitCode(result.ok() ? 0 : 1)
                  .build();
          response.writeDelimitedTo(System.out);
        }
        System.out.flush();
      } catch (IOException e) {
        e.printStackTrace();
        return 1;
      }
    }
    return 0;
  }

  /** Return result of a {@link VanillaJavaBuilder} build. */
  public static class VanillaJavaBuilderResult {
    private final boolean ok;
    private final String output;

    public VanillaJavaBuilderResult(boolean ok, String output) {
      this.ok = ok;
      this.output = output;
    }

    /** True if the compilation was succesfull. */
    public boolean ok() {
      return ok;
    }

    /** Log output from the compilation. */
    public String output() {
      return output;
    }
  }

  public VanillaJavaBuilderResult run(List<String> args) throws IOException {
    OptionsParser optionsParser;
    try {
      optionsParser = new OptionsParser(args);
    } catch (InvalidCommandLineException e) {
      return new VanillaJavaBuilderResult(false, e.getMessage());
    }
    DiagnosticCollector<JavaFileObject> diagnosticCollector = new DiagnosticCollector<>();
    StringWriter output = new StringWriter();
    JavaCompiler javaCompiler = ToolProvider.getSystemJavaCompiler();
    Path tempDir = Paths.get(firstNonNull(optionsParser.getTempDir(), "_tmp"));
    Path nativeHeaderDir = tempDir.resolve("native_headers");
    Files.createDirectories(nativeHeaderDir);
    boolean ok;
    try (StandardJavaFileManager fileManager =
        javaCompiler.getStandardFileManager(diagnosticCollector, ENGLISH, UTF_8)) {
      setLocations(optionsParser, fileManager, nativeHeaderDir);
      ImmutableList<JavaFileObject> sources = getSources(optionsParser, fileManager);
      if (sources.isEmpty()) {
        ok = true;
      } else {
        CompilationTask task =
            javaCompiler.getTask(
                new PrintWriter(output, true),
                fileManager,
                diagnosticCollector,
                JavacOptions.removeBazelSpecificFlags(optionsParser.getJavacOpts()),
                ImmutableList.<String>of() /*classes*/,
                sources);
        setProcessors(optionsParser, fileManager, task);
        ok = task.call();
      }
    }
    if (ok) {
      writeOutput(optionsParser);
      writeNativeHeaderOutput(optionsParser, nativeHeaderDir);
    }
    writeGeneratedSourceOutput(optionsParser);
    // the jdeps output doesn't include any information about dependencies, but Bazel still expects
    // the file to be created
    if (optionsParser.getOutputDepsProtoFile() != null) {
      try (OutputStream os =
          Files.newOutputStream(Paths.get(optionsParser.getOutputDepsProtoFile()))) {
        Deps.Dependencies.newBuilder()
            .setRuleLabel(optionsParser.getTargetLabel())
            .setSuccess(ok)
            .build()
            .writeTo(os);
      }
    }
    // TODO(cushon): support manifest protos & genjar
    if (optionsParser.getManifestProtoPath() != null) {
      try (OutputStream os =
          Files.newOutputStream(Paths.get(optionsParser.getManifestProtoPath()))) {
        Manifest.getDefaultInstance().writeTo(os);
      }
    }

    for (Diagnostic<? extends JavaFileObject> diagnostic : diagnosticCollector.getDiagnostics()) {
      String code = diagnostic.getCode();
      if (code.startsWith("compiler.note.deprecated")
          || code.startsWith("compiler.note.unchecked")
          || code.equals("compiler.warn.sun.proprietary")) {
        continue;
      }
      StringBuilder message = new StringBuilder();
      if (diagnostic.getSource() != null) {
        message.append(diagnostic.getSource().getName());
        if (diagnostic.getLineNumber() != -1) {
          message.append(':').append(diagnostic.getLineNumber());
        }
        message.append(": ");
      }
      message.append(diagnostic.getKind().toString().toLowerCase(ENGLISH));
      message.append(": ").append(diagnostic.getMessage(ENGLISH)).append(System.lineSeparator());
      output.write(message.toString());
    }
    return new VanillaJavaBuilderResult(ok, output.toString());
  }

  /** Returns the sources to compile, including any source jar entries. */
  private ImmutableList<JavaFileObject> getSources(
      OptionsParser optionsParser, StandardJavaFileManager fileManager) throws IOException {
    final ImmutableList.Builder<JavaFileObject> sources = ImmutableList.builder();
    sources.addAll(fileManager.getJavaFileObjectsFromStrings(optionsParser.getSourceFiles()));
    for (String sourceJar : optionsParser.getSourceJars()) {
      for (final Path root : getJarFileSystem(Paths.get(sourceJar)).getRootDirectories()) {
        Files.walkFileTree(
            root,
            new SimpleFileVisitor<Path>() {
              @Override
              public FileVisitResult visitFile(Path path, BasicFileAttributes attrs)
                  throws IOException {
                if (path.getFileName().toString().endsWith(".java")) {
                  sources.add(new SourceJarFileObject(root, path));
                }
                return FileVisitResult.CONTINUE;
              }
            });
      }
    }
    return sources.build();
  }

  /** Sets the compilation search paths and output directories. */
  private static void setLocations(
      OptionsParser optionsParser, StandardJavaFileManager fileManager, Path nativeHeaderDir)
      throws IOException {
    fileManager.setLocation(StandardLocation.CLASS_PATH, toFiles(optionsParser.getClassPath()));
    Iterable<File> bootClassPath = toFiles(optionsParser.getBootClassPath());
    // The bootclasspath may legitimately be empty if --release is being used.
    if (!Iterables.isEmpty(bootClassPath)) {
      fileManager.setLocation(StandardLocation.PLATFORM_CLASS_PATH, bootClassPath);
    }
    fileManager.setLocation(
        StandardLocation.ANNOTATION_PROCESSOR_PATH, toFiles(optionsParser.getProcessorPath()));
    if (optionsParser.getSourceGenDir() != null) {
      setOutputLocation(
          fileManager, StandardLocation.SOURCE_OUTPUT, Paths.get(optionsParser.getSourceGenDir()));
    }
    if (optionsParser.getNativeHeaderOutput() != null) {
      setOutputLocation(fileManager, StandardLocation.NATIVE_HEADER_OUTPUT, nativeHeaderDir);
    }
    setOutputLocation(
        fileManager, StandardLocation.CLASS_OUTPUT, Paths.get(optionsParser.getClassDir()));
  }

  private static void setOutputLocation(
      StandardJavaFileManager fileManager, StandardLocation location, Path path)
      throws IOException {
    createOutputDirectory(path);
    fileManager.setLocation(location, ImmutableList.of(path.toFile()));
  }

  /** Sets the compilation's annotation processors. */
  private static void setProcessors(
      OptionsParser optionsParser, StandardJavaFileManager fileManager, CompilationTask task) {
    ClassLoader processorLoader =
        fileManager.getClassLoader(StandardLocation.ANNOTATION_PROCESSOR_PATH);
    ImmutableList.Builder<Processor> processors = ImmutableList.builder();
    for (String processor : optionsParser.getProcessorNames()) {
      try {
        processors.add(
            (Processor) processorLoader.loadClass(processor).getConstructor().newInstance());
      } catch (ReflectiveOperationException e) {
        throw new LinkageError(e.getMessage(), e);
      }
    }
    task.setProcessors(processors.build());
  }

  /** Writes a jar containing any sources generated by annotation processors. */
  private static void writeGeneratedSourceOutput(OptionsParser optionsParser) throws IOException {
    if (optionsParser.getGeneratedSourcesOutputJar() == null) {
      return;
    }
    JarCreator jar = new JarCreator(optionsParser.getGeneratedSourcesOutputJar());
    jar.setNormalize(true);
    jar.setCompression(optionsParser.compressJar());
    jar.addDirectory(optionsParser.getSourceGenDir());
    jar.execute();
  }

  private static void writeNativeHeaderOutput(OptionsParser optionsParser, Path nativeHeaderDir)
      throws IOException {
    if (optionsParser.getNativeHeaderOutput() == null) {
      return;
    }
    JarCreator jar = new JarCreator(optionsParser.getNativeHeaderOutput());
    try {
      jar.setNormalize(true);
      jar.setCompression(optionsParser.compressJar());
      jar.addDirectory(nativeHeaderDir);
    } finally {
      jar.execute();
    }
  }

  /** Writes the class output jar, including any resource entries. */
  private static void writeOutput(OptionsParser optionsParser) throws IOException {
    JarCreator jar = new JarCreator(optionsParser.getOutputJar());
    jar.setNormalize(true);
    jar.setCompression(optionsParser.compressJar());
    jar.addDirectory(optionsParser.getClassDir());
    jar.execute();
  }

  private static ImmutableList<File> toFiles(List<String> classPath) {
    if (classPath == null) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<File> files = ImmutableList.builder();
    for (String path : classPath) {
      files.add(new File(path));
    }
    return files.build();
  }

  @Override
  public void close() throws IOException {
    for (FileSystem fs : filesystems.values()) {
      fs.close();
    }
  }

  /**
   * Wraps a {@link Path} as a {@link JavaFileObject}; used to avoid extracting source jar entries
   * to disk when using file managers that don't support nio.
   */
  private static class SourceJarFileObject extends SimpleJavaFileObject {
    private final Path path;

    public SourceJarFileObject(Path root, Path path) {
      super(URI.create("file:/" + root + "!" + root.resolve(path)), Kind.SOURCE);
      this.path = path;
    }

    @Override
    public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException {
      return new String(Files.readAllBytes(path), UTF_8);
    }
  }

  private static void createOutputDirectory(Path dir) throws IOException {
    if (Files.exists(dir)) {
      try {
        MoreFiles.deleteRecursively(dir, RecursiveDeleteOption.ALLOW_INSECURE);
      } catch (IOException e) {
        throw new IOException("Cannot clean output directory '" + dir + "'", e);
      }
    }
    Files.createDirectories(dir);
  }
}
