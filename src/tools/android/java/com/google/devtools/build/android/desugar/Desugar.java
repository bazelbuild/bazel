// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Closer;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.errorprone.annotations.MustBeClosed;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;

/**
 * Command-line tool to desugar Java 8 constructs that dx doesn't know what to do with, in
 * particular lambdas and method references.
 */
class Desugar {

  /** Commandline options for {@link Desugar}. */
  public static class Options extends OptionsBase {
    @Option(
      name = "input",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help =
        "Input Jar or directory with classes to desugar (required, the n-th input is paired with"
        + "the n-th output)."
    )
    public List<Path> inputJars;

    @Option(
      name = "classpath_entry",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      converter = ExistingPathConverter.class,
      help = "Ordered classpath to resolve symbols in the --input Jar, like javac's -cp flag."
    )
    public List<Path> classpath;

    @Option(
      name = "bootclasspath_entry",
      allowMultiple = true,
      defaultValue = "",
      category = "input",
      converter = ExistingPathConverter.class,
      help = "Bootclasspath that was used to compile the --input Jar with, like javac's "
          + "-bootclasspath flag (required)."
    )
    public List<Path> bootclasspath;

    @Option(
      name = "allow_empty_bootclasspath",
      defaultValue = "false",
      category = "undocumented"
    )
    public boolean allowEmptyBootclasspath;

    @Option(
      name = "only_desugar_javac9_for_lint",
      defaultValue = "false",
      help =
          "A temporary flag specifically for android lint, subject to removal anytime (DO NOT USE)",
      category = "undocumented"
    )
    public boolean onlyDesugarJavac9ForLint;

    @Option(
      name = "rewrite_calls_to_long_compare",
      defaultValue = "true",
      help = "rewrite calls to Long.compare(long, long) to the JVM instruction lcmp",
      category = "misc"
    )
    public boolean enableRewritingOfLongCompare;

    @Option(
      name = "output",
      allowMultiple = true,
      defaultValue = "",
      category = "output",
      converter = PathConverter.class,
      abbrev = 'o',
      help =
          "Output Jar or directory to write desugared classes into (required, the n-th output is "
              + "paired with the n-th input, output must be a Jar if input is a Jar)."
    )
    public List<Path> outputJars;

    @Option(
      name = "verbose",
      defaultValue = "false",
      category = "misc",
      abbrev = 'v',
      help = "Enables verbose debugging output."
    )
    public boolean verbose;

    @Option(
      name = "min_sdk_version",
      defaultValue = "1",
      category = "misc",
      help = "Minimum targeted sdk version.  If >= 24, enables default methods in interfaces."
    )
    public int minSdkVersion;

    @Option(
      name = "copy_bridges_from_classpath",
      defaultValue = "false",
      category = "misc",
      help = "Copy bridges from classpath to desugared classes."
    )
    public boolean copyBridgesFromClasspath;

    @Option(
      name = "core_library",
      defaultValue = "false",
      category = "undocumented",
      implicitRequirements = "--allow_empty_bootclasspath",
      help = "Enables rewriting to desugar java.* classes."
    )
    public boolean coreLibrary;
  }

  private final Options options;
  private final Path dumpDirectory;
  private final CoreLibraryRewriter rewriter;
  private final LambdaClassMaker lambdas;
  private final boolean allowDefaultMethods;
  private final boolean allowCallsToObjectsNonNull;

  private Desugar(Options options, Path dumpDirectory) {
    this.options = options;
    this.dumpDirectory = dumpDirectory;
    this.rewriter = new CoreLibraryRewriter(options.coreLibrary ? "__desugar__/" : "");
    this.lambdas = new LambdaClassMaker(dumpDirectory);
    this.allowDefaultMethods = options.minSdkVersion >= 24;
    this.allowCallsToObjectsNonNull = options.minSdkVersion >= 19;
  }

  public void desugar() throws Exception {
    try (Closer closer = Closer.create()) {
      IndexedInputs indexedClasspath =
          new IndexedInputs(toRegisteredInputFileProvider(closer, options.classpath));
      // Use a classloader that as much as possible uses the provided bootclasspath instead of
      // the tool's system classloader.  Unfortunately we can't do that for java. classes.
      ClassLoader bootclassloader =
          options.bootclasspath.isEmpty()
              ? new ThrowingClassLoader()
              : new HeaderClassLoader(
                  new IndexedInputs(toRegisteredInputFileProvider(closer, options.bootclasspath)),
                  rewriter,
                  new ThrowingClassLoader());

      // Process each input separately
      for (InputOutputPair inputOutputPair : toInputOutputPairs(options)) {
        Path inputPath = inputOutputPair.getInput();
        Path outputPath = inputOutputPair.getOutput();
        checkState(
            Files.isDirectory(inputPath) || !Files.isDirectory(outputPath),
            "Input jar file requires an output jar file");

        try (OutputFileProvider outputFileProvider = toOutputFileProvider(outputPath);
            InputFileProvider inputFiles = toInputFileProvider(inputPath)) {
          IndexedInputs indexedInputFiles = new IndexedInputs(ImmutableList.of(inputFiles));
          // Prepend classpath with input file itself so LambdaDesugaring can load classes with
          // lambdas.
          IndexedInputs indexedClasspathAndInputFiles =
              indexedClasspath.withParent(indexedInputFiles);
          // Note that input file and classpath need to be in the same classloader because
          // we typically get the header Jar for inputJar on the classpath and having the header
          // Jar in a parent loader means the header version is preferred over the real thing.
          ClassLoader loader =
              new HeaderClassLoader(indexedClasspathAndInputFiles, rewriter, bootclassloader);

          ClassReaderFactory readerFactory =
              new ClassReaderFactory(
                  (options.copyBridgesFromClasspath && !allowDefaultMethods)
                      ? indexedClasspathAndInputFiles
                      : indexedInputFiles,
                  rewriter);

          ImmutableSet.Builder<String> interfaceLambdaMethodCollector = ImmutableSet.builder();

          // Process inputs, desugaring as we go
          for (String filename : inputFiles) {
            try (InputStream content = inputFiles.getInputStream(filename)) {
              // We can write classes uncompressed since they need to be converted to .dex format
              // for Android anyways. Resources are written as they were in the input jar to avoid
              // any danger of accidentally uncompressed resources ending up in an .apk.
              if (filename.endsWith(".class")) {
                ClassReader reader = rewriter.reader(content);
                CoreLibraryRewriter.UnprefixingClassWriter writer =
                    rewriter.writer(ClassWriter.COMPUTE_MAXS /*for bridge methods*/);
                ClassVisitor visitor = writer;

                if (!options.onlyDesugarJavac9ForLint) {
                  if (!allowDefaultMethods) {
                    visitor = new Java7Compatibility(visitor, readerFactory);
                  }

                  visitor =
                      new LambdaDesugaring(
                          visitor,
                          loader,
                          lambdas,
                          interfaceLambdaMethodCollector,
                          allowDefaultMethods);
                }

                if (!allowCallsToObjectsNonNull) {
                  visitor = new ObjectsRequireNonNullMethodRewriter(visitor);
                }
                if (options.enableRewritingOfLongCompare) {
                  visitor = new LongCompareMethodRewriter(visitor);
                }
                reader.accept(visitor, 0);

                outputFileProvider.write(filename, writer.toByteArray());
              } else {
                outputFileProvider.copyFrom(filename, inputFiles);
              }
            }
          }

          ImmutableSet<String> interfaceLambdaMethods = interfaceLambdaMethodCollector.build();
          checkState(
              !allowDefaultMethods || interfaceLambdaMethods.isEmpty(),
              "Desugaring with default methods enabled moved interface lambdas");

          // Write out the lambda classes we generated along the way
          ImmutableMap<Path, LambdaInfo> lambdaClasses = lambdas.drain();
          checkState(
              !options.onlyDesugarJavac9ForLint || lambdaClasses.isEmpty(),
              "There should be no lambda classes generated: %s",
              lambdaClasses.keySet());

          for (Map.Entry<Path, LambdaInfo> lambdaClass : lambdaClasses.entrySet()) {
            try (InputStream bytecode =
                Files.newInputStream(dumpDirectory.resolve(lambdaClass.getKey()))) {
              ClassReader reader = rewriter.reader(bytecode);
              CoreLibraryRewriter.UnprefixingClassWriter writer =
                  rewriter.writer(ClassWriter.COMPUTE_MAXS /*for invoking bridges*/);
              ClassVisitor visitor = writer;

              if (!allowDefaultMethods) {
                // null ClassReaderFactory b/c we don't expect to need it for lambda classes
                visitor = new Java7Compatibility(visitor, (ClassReaderFactory) null);
              }

              visitor =
                  new LambdaClassFixer(
                      visitor,
                      lambdaClass.getValue(),
                      readerFactory,
                      interfaceLambdaMethods,
                      allowDefaultMethods);
              // Send lambda classes through desugaring to make sure there's no invokedynamic
              // instructions in generated lambda classes (checkState below will fail)
              visitor = new LambdaDesugaring(visitor, loader, lambdas, null, allowDefaultMethods);
              if (!allowCallsToObjectsNonNull) {
                // Not sure whether there will be implicit null check emitted by javac, so we rerun
                // the inliner again
                visitor = new ObjectsRequireNonNullMethodRewriter(visitor);
              }
              if (options.enableRewritingOfLongCompare) {
                visitor = new LongCompareMethodRewriter(visitor);
              }
              reader.accept(visitor, 0);
              String filename =
                  rewriter.unprefix(lambdaClass.getValue().desiredInternalName()) + ".class";
              outputFileProvider.write(filename, writer.toByteArray());
            }
          }
        }

        ImmutableMap<Path, LambdaInfo> leftBehind = lambdas.drain();
        checkState(leftBehind.isEmpty(), "Didn't process %s", leftBehind);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    // It is important that this method is called first. See its javadoc.
    Path dumpDirectory = createAndRegisterLambdaDumpDirectory();
    Options options = parseCommandLineOptions(args);
    if (options.verbose) {
      System.out.printf("Lambda classes will be written under %s%n", dumpDirectory);
    }
    new Desugar(options, dumpDirectory).desugar();
  }

  /**
   * LambdaClassMaker generates lambda classes for us, but it does so by essentially simulating the
   * call to LambdaMetafactory that the JVM would make when encountering an invokedynamic.
   * LambdaMetafactory is in the JDK and its implementation has a property to write out ("dump")
   * generated classes, which we take advantage of here. Set property before doing anything else
   * since the property is read in the static initializer; if this breaks we can investigate setting
   * the property when calling the tool.
   */
  private static Path createAndRegisterLambdaDumpDirectory() throws IOException {
    Path dumpDirectory = Files.createTempDirectory("lambdas");
    System.setProperty(
        LambdaClassMaker.LAMBDA_METAFACTORY_DUMPER_PROPERTY, dumpDirectory.toString());

    deleteTreeOnExit(dumpDirectory);
    return dumpDirectory;
  }

  private static Options parseCommandLineOptions(String[] args) throws IOException {
    if (args.length == 1 && args[0].startsWith("@")) {
      args = Files.readAllLines(Paths.get(args[0].substring(1)), ISO_8859_1).toArray(new String[0]);
    }

    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.setAllowResidue(false);
    optionsParser.parseAndExitUponError(args);

    Options options = optionsParser.getOptions(Options.class);

    checkArgument(!options.inputJars.isEmpty(), "--input is required");
    checkArgument(
        options.inputJars.size() == options.outputJars.size(),
        "Desugar requires the same number of inputs and outputs to pair them. #input=%s,#output=%s",
        options.inputJars.size(),
        options.outputJars.size());
    checkArgument(
        !options.bootclasspath.isEmpty() || options.allowEmptyBootclasspath,
        "At least one --bootclasspath_entry is required");
    for (Path path : options.classpath) {
      checkArgument(!Files.isDirectory(path), "Classpath entry must be a jar file: %s", path);
    }
    for (Path path : options.bootclasspath) {
      checkArgument(!Files.isDirectory(path), "Bootclasspath entry must be a jar file: %s", path);
    }
    return options;
  }

  private static ImmutableList<InputOutputPair> toInputOutputPairs(Options options) {
    final ImmutableList.Builder<InputOutputPair> ioPairListbuilder = ImmutableList.builder();
    for (Iterator<Path> inputIt = options.inputJars.iterator(),
                outputIt = options.outputJars.iterator();
                inputIt.hasNext();) {
      ioPairListbuilder.add(InputOutputPair.create(inputIt.next(), outputIt.next()));
    }
    return ioPairListbuilder.build();
  }

  private static class ThrowingClassLoader extends ClassLoader {
    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      if (name.startsWith("java.")) {
        // Use system class loader for java. classes, since ClassLoader.defineClass gets
        // grumpy when those don't come from the standard place.
        return super.loadClass(name, resolve);
      }
      throw new ClassNotFoundException();
    }
  }

  private static void deleteTreeOnExit(final Path directory) {
    Thread shutdownHook =
        new Thread() {
          @Override
          public void run() {
            try {
              deleteTree(directory);
            } catch (IOException e) {
              throw new RuntimeException("Failed to delete " + directory, e);
            }
          }
        };
    Runtime.getRuntime().addShutdownHook(shutdownHook);
  }

  /** Recursively delete a directory. */
  private static void deleteTree(final Path directory) throws IOException {
    if (directory.toFile().exists()) {
      Files.walkFileTree(
          directory,
          new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs)
                throws IOException {
              Files.delete(file);
              return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc)
                throws IOException {
              Files.delete(dir);
              return FileVisitResult.CONTINUE;
            }
          });
    }
  }

  /** Transform a Path to an {@link OutputFileProvider} */
  @MustBeClosed
  private static OutputFileProvider toOutputFileProvider(Path path)
      throws IOException {
    if (Files.isDirectory(path)) {
      return new DirectoryOutputFileProvider(path);
    } else {
      return new ZipOutputFileProvider(path);
    }
  }

  /** Transform a Path to an InputFileProvider that needs to be closed by the caller. */
  @MustBeClosed
  private static InputFileProvider toInputFileProvider(Path path)
      throws IOException {
    if (Files.isDirectory(path)) {
      return new DirectoryInputFileProvider(path);
    } else {
      return new ZipInputFileProvider(path);
    }
  }

  /**
   * Transform a list of Path to a list of ZipInputFileProvider and register them with the given
   * closer.
   */
  private static ImmutableList<InputFileProvider> toRegisteredInputFileProvider(
      Closer closer, List<Path> paths) throws IOException {
    ImmutableList.Builder<InputFileProvider> builder = new ImmutableList.Builder<>();
    for (Path path : paths) {
      checkState(!Files.isDirectory(path), "Directory is not supported: %s", path);
      builder.add(closer.register(new ZipInputFileProvider(path)));
    }
    return builder.build();
  }

  /**
   * Pair input and output.
   */
  @AutoValue
  abstract static class InputOutputPair {

    static InputOutputPair create(Path input, Path output) {
      return new AutoValue_Desugar_InputOutputPair(input, output);
    }

    abstract Path getInput();

    abstract Path getOutput();
  }
}
