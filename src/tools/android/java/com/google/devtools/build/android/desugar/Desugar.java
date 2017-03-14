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

import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
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
      defaultValue = "null",
      category = "input",
      converter = ExistingPathConverter.class,
      abbrev = 'i',
      help = "Input Jar with classes to desugar (required)."
    )
    public Path inputJar;

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
      name = "output",
      defaultValue = "null",
      category = "output",
      converter = PathConverter.class,
      abbrev = 'o',
      help = "Output Jar to write desugared classes into (required)."
    )
    public Path outputJar;

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

  public static void main(String[] args) throws Exception {
    // LambdaClassMaker generates lambda classes for us, but it does so by essentially simulating
    // the call to LambdaMetafactory that the JVM would make when encountering an invokedynamic.
    // LambdaMetafactory is in the JDK and its implementation has a property to write out ("dump")
    // generated classes, which we take advantage of here.  Set property before doing anything else
    // since the property is read in the static initializer; if this breaks we can investigate
    // setting the property when calling the tool.
    Path dumpDirectory = Files.createTempDirectory("lambdas");
    System.setProperty(
        LambdaClassMaker.LAMBDA_METAFACTORY_DUMPER_PROPERTY, dumpDirectory.toString());

    deleteTreeOnExit(dumpDirectory);

    if (args.length == 1 && args[0].startsWith("@")) {
      args = Files.readAllLines(Paths.get(args[0].substring(1)), ISO_8859_1).toArray(new String[0]);
    }

    OptionsParser optionsParser = OptionsParser.newOptionsParser(Options.class);
    optionsParser.setAllowResidue(false);
    optionsParser.parseAndExitUponError(args);
    Options options = optionsParser.getOptions(Options.class);

    checkState(options.inputJar != null, "--input is required");
    checkState(options.outputJar != null, "--output is required");
    checkState(!options.bootclasspath.isEmpty() || options.allowEmptyBootclasspath,
        "Bootclasspath required to desugar %s", options.inputJar);

    if (options.verbose) {
      System.out.printf("Lambda classes will be written under %s%n", dumpDirectory);
    }

    CoreLibraryRewriter rewriter =
        new CoreLibraryRewriter(options.coreLibrary ? "__desugar__/" : "");

    IndexedJars appIndexedJar = new IndexedJars(ImmutableList.of(options.inputJar));
    IndexedJars appAndClasspathIndexedJars = new IndexedJars(options.classpath, appIndexedJar);
    ClassLoader loader =
        createClassLoader(rewriter, options.bootclasspath, appAndClasspathIndexedJars);
    boolean allowDefaultMethods = options.minSdkVersion >= 24;
    boolean allowCallsToObjectsNonNull = options.minSdkVersion >= 19;
    try (ZipFile in = new ZipFile(options.inputJar.toFile());
        ZipOutputStream out =
            new ZipOutputStream(
                new BufferedOutputStream(Files.newOutputStream(options.outputJar)))) {
      LambdaClassMaker lambdas = new LambdaClassMaker(dumpDirectory);
      ClassReaderFactory readerFactory =
          new ClassReaderFactory(
              (options.copyBridgesFromClasspath && !allowDefaultMethods)
                  ? appAndClasspathIndexedJars
                  : appIndexedJar,
              rewriter);

      ImmutableSet.Builder<String> interfaceLambdaMethodCollector = ImmutableSet.builder();

      // Process input Jar, desugaring as we go
      for (Enumeration<? extends ZipEntry> entries = in.entries(); entries.hasMoreElements(); ) {
        ZipEntry entry = entries.nextElement();
        try (InputStream content = in.getInputStream(entry)) {
          // We can write classes uncompressed since they need to be converted to .dex format for
          // Android anyways. Resources are written as they were in the input jar to avoid any
          // danger of accidentally uncompressed resources ending up in an .apk.
          if (entry.getName().endsWith(".class")) {
            ClassReader reader = rewriter.reader(content);
            CoreLibraryRewriter.UnprefixingClassWriter writer =
                rewriter.writer(ClassWriter.COMPUTE_MAXS /*for bridge methods*/);
            ClassVisitor visitor = writer;
            if (!allowDefaultMethods) {
              visitor = new Java7Compatibility(visitor, readerFactory);
            }

            visitor =
                new LambdaDesugaring(
                    visitor, loader, lambdas, interfaceLambdaMethodCollector, allowDefaultMethods);

            if (!allowCallsToObjectsNonNull) {
              visitor = new ObjectsRequireNonNullMethodInliner(visitor);
            }
            reader.accept(visitor, 0);

            writeStoredEntry(out, entry.getName(), writer.toByteArray());
          } else {
            // TODO(bazel-team): Avoid de- and re-compressing resource files
            ZipEntry destEntry = new ZipEntry(entry);
            destEntry.setCompressedSize(-1);
            out.putNextEntry(destEntry);
            ByteStreams.copy(content, out);
            out.closeEntry();
          }
        }
      }

      ImmutableSet<String> interfaceLambdaMethods = interfaceLambdaMethodCollector.build();
      if (allowDefaultMethods) {
        checkState(
            interfaceLambdaMethods.isEmpty(),
            "Desugaring with default methods enabled moved interface lambdas");
      }

      // Write out the lambda classes we generated along the way
      for (Map.Entry<Path, LambdaInfo> lambdaClass : lambdas.drain().entrySet()) {
        try (InputStream bytecode = Files.newInputStream(lambdaClass.getKey())) {
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
            // Not sure whether there will be implicit null check emitted by javac, so we rerun the
            // inliner again
            visitor = new ObjectsRequireNonNullMethodInliner(visitor);
          }
          reader.accept(visitor, 0);
          String filename =
              rewriter.unprefix(lambdaClass.getValue().desiredInternalName()) + ".class";
          writeStoredEntry(out, filename, writer.toByteArray());
        }
      }

      Map<Path, LambdaInfo> leftBehind = lambdas.drain();
      checkState(leftBehind.isEmpty(), "Didn't process %s", leftBehind);
    }
  }

  private static void writeStoredEntry(ZipOutputStream out, String filename, byte[] content)
      throws IOException {
    // Need to pre-compute checksum for STORED (uncompressed) entries)
    CRC32 checksum = new CRC32();
    checksum.update(content);

    ZipEntry result = new ZipEntry(filename);
    result.setTime(0L); // Use stable timestamp Jan 1 1980
    result.setCrc(checksum.getValue());
    result.setSize(content.length);
    result.setCompressedSize(content.length);
    // Write uncompressed, since this is just an intermediary artifact that we will convert to .dex
    result.setMethod(ZipEntry.STORED);

    out.putNextEntry(result);
    out.write(content);
    out.closeEntry();
  }

  private static ClassLoader createClassLoader(CoreLibraryRewriter rewriter,
      List<Path> bootclasspath, IndexedJars appAndClasspathIndexedJars) throws IOException {
    // Use a classloader that as much as possible uses the provided bootclasspath instead of
    // the tool's system classloader.  Unfortunately we can't do that for java. classes.
    ClassLoader parent = new ThrowingClassLoader();
    if (!bootclasspath.isEmpty()) {
      parent = new HeaderClassLoader(new IndexedJars(bootclasspath), rewriter, parent);
    }
    // Prepend classpath with input jar itself so LambdaDesugaring can load classes with lambdas.
    // Note that inputJar and classpath need to be in the same classloader because we typically get
    // the header Jar for inputJar on the classpath and having the header Jar in a parent loader
    // means the header version is preferred over the real thing.
    return new HeaderClassLoader(appAndClasspathIndexedJars, rewriter, parent);
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
}
