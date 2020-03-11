// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.scan;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closer;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import com.google.devtools.build.android.desugar.io.HeaderClassLoader;
import com.google.devtools.build.android.desugar.io.IndexedInputs;
import com.google.devtools.build.android.desugar.io.InputFileProvider;
import com.google.devtools.build.android.desugar.io.ThrowingClassLoader;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.Type;

class KeepScanner {

  public static class KeepScannerOptions extends OptionsBase {
    @Option(
        name = "input",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = OptionEffectTag.UNKNOWN,
        converter = ExistingPathConverter.class,
        abbrev = 'i',
        help = "Input Jar with classes to scan.")
    public Path inputJars;

    @Option(
        name = "classpath_entry",
        allowMultiple = true,
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help =
            "Ordered classpath (Jar or directory) to resolve symbols in the --input Jar, like "
                + "javac's -cp flag.")
    public List<Path> classpath;

    @Option(
        name = "bootclasspath_entry",
        allowMultiple = true,
        defaultValue = "",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        converter = ExistingPathConverter.class,
        help =
            "Bootclasspath that was used to compile the --input Jar with, like javac's "
                + "-bootclasspath flag (required).")
    public List<Path> bootclasspath;

    @Option(
        name = "keep_file",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = OptionEffectTag.UNKNOWN,
        converter = PathConverter.class,
        help = "Where to write keep rules to.")
    public Path keepDest;

    @Option(
        name = "prefix",
        defaultValue = "j$/",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = OptionEffectTag.UNKNOWN,
        help = "type to scan for.")
    public String prefix;
  }

  public static void main(String... args) throws Exception {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(KeepScannerOptions.class)
            .allowResidue(false)
            .argsPreProcessor(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()))
            .build();
    parser.parseAndExitUponError(args);
    KeepScannerOptions options = parser.getOptions(KeepScannerOptions.class);

    Map<String, ImmutableSet<KeepReference>> seeds;
    try (Closer closer = Closer.create()) {
      // TODO(kmb): Try to share more of this code with Desugar binary
      IndexedInputs classpath =
          new IndexedInputs(toRegisteredInputFileProvider(closer, options.classpath));
      IndexedInputs bootclasspath =
          new IndexedInputs(toRegisteredInputFileProvider(closer, options.bootclasspath));

      // Construct classloader from classpath.  Since we're assuming the prefix we're looking for
      // isn't part of the input itself we shouldn't need to include the input in the classloader.
      CoreLibraryRewriter noopRewriter = new CoreLibraryRewriter("");
      ClassLoader classloader =
          new HeaderClassLoader(
              classpath,
              noopRewriter,
              new HeaderClassLoader(bootclasspath, noopRewriter, new ThrowingClassLoader()));
      seeds = scan(checkNotNull(options.inputJars), options.prefix, classloader);
    }

    try (PrintStream out =
        new PrintStream(
            Files.newOutputStream(options.keepDest, CREATE), /*autoFlush=*/ false, "UTF-8")) {
      writeKeepDirectives(out, seeds);
    }
  }

  /**
   * Writes a -keep rule for each class listing any members to keep. We sort classes and members so
   * the output is deterministic.
   */
  private static void writeKeepDirectives(
      PrintStream out, Map<String, ImmutableSet<KeepReference>> seeds) {
    seeds.entrySet().stream()
        .sorted(comparing(Map.Entry::getKey))
        .forEachOrdered(
            type -> {
              out.printf("-keep class %s {%n", type.getKey().replace('/', '.'));
              type.getValue().stream()
                  .filter(KeepReference::isMemberReference)
                  .sorted(comparing(KeepReference::name).thenComparing(KeepReference::desc))
                  .map(ref -> toKeepDescriptor(ref))
                  .distinct() // drop duplicates due to method descriptors with different returns
                  .forEachOrdered(line -> out.append("  ").append(line).append(";").println());
              out.printf("}%n");
            });
  }

  /** Scans for and returns references with owners matching the given prefix grouped by owner. */
  private static Map<String, ImmutableSet<KeepReference>> scan(
      Path jarFile, String prefix, ClassLoader classpath) throws IOException {
    // We read the Jar sequentially since ZipFile uses locks anyway but then allow scanning each
    // class in parallel.
    try (ZipFile zip = new ZipFile(jarFile.toFile())) {
      return zip.stream()
          .filter(entry -> entry.getName().endsWith(".class"))
          .map(entry -> readFully(zip, entry))
          .parallel()
          .flatMap(
              content -> PrefixReferenceScanner.scan(new ClassReader(content), prefix).stream())
          .distinct() // so we don't process the same reference multiple times next
          .map(ref -> nearestDeclaration(ref, classpath))
          .collect(
              Collectors.groupingByConcurrent(
                  KeepReference::internalName, ImmutableSet.toImmutableSet()));
    }
  }

  private static byte[] readFully(ZipFile zip, ZipEntry entry) {
    byte[] result = new byte[(int) entry.getSize()];
    try (InputStream content = zip.getInputStream(entry)) {
      ByteStreams.readFully(content, result);
      return result;
    } catch (IOException e) {
      throw new IOError(e);
    }
  }

  /**
   * Find the nearest definition of the given reference in the class hierarchy and return the
   * modified reference. This is needed b/c bytecode sometimes refers to a method or field using an
   * owner type that inherits the method or field instead of defining the member itself. In that
   * case we need to find and keep the inherited definition.
   */
  private static KeepReference nearestDeclaration(KeepReference ref, ClassLoader classpath) {
    if (!ref.isMemberReference() || "<init>".equals(ref.name())) {
      return ref; // class and constructor references don't need any further work
    }

    Class<?> clazz;
    try {
      clazz = classpath.loadClass(ref.internalName().replace('/', '.'));
    } catch (ClassNotFoundException e) {
      throw (NoClassDefFoundError) new NoClassDefFoundError("Couldn't load " + ref).initCause(e);
    }

    Class<?> owner = findDeclaringClass(clazz, ref);
    if (owner == clazz) {
      return ref;
    }
    String parent = checkNotNull(owner, "Can't resolve: %s", ref).getName().replace('.', '/');
    return KeepReference.memberReference(parent, ref.name(), ref.desc());
  }

  private static Class<?> findDeclaringClass(Class<?> clazz, KeepReference ref) {
    if (ref.isFieldReference()) {
      try {
        return clazz.getField(ref.name()).getDeclaringClass();
      } catch (NoSuchFieldException e) {
        // field must be non-public, so search class hierarchy
        do {
          try {
            return clazz.getDeclaredField(ref.name()).getDeclaringClass();
          } catch (NoSuchFieldException ignored) {
            // fall through for clarity
          }
          clazz = clazz.getSuperclass();
        } while (clazz != null);
      }
    } else {
      checkState(ref.isMethodReference());
      Type descriptor = Type.getMethodType(ref.desc());
      for (Method m : clazz.getMethods()) {
        if (m.getName().equals(ref.name()) && Type.getType(m).equals(descriptor)) {
          return m.getDeclaringClass();
        }
      }
      do {
        // Method must be non-public, so search class hierarchy
        for (Method m : clazz.getDeclaredMethods()) {
          if (m.getName().equals(ref.name()) && Type.getType(m).equals(descriptor)) {
            return m.getDeclaringClass();
          }
        }
        clazz = clazz.getSuperclass();
      } while (clazz != null);
    }
    return null;
  }

  private static CharSequence toKeepDescriptor(KeepReference member) {
    StringBuilder result = new StringBuilder();
    if (member.isMethodReference()) {
      if (!"<init>".equals(member.name())) {
        result.append("*** ");
      }
      result.append(member.name()).append("(");
      // Ignore return type as it's unique in the source language
      boolean first = true;
      for (Type param : Type.getMethodType(member.desc()).getArgumentTypes()) {
        if (first) {
          first = false;
        } else {
          result.append(", ");
        }
        result.append(param.getClassName());
      }
      result.append(")");
    } else {
      checkArgument(member.isFieldReference());
      result.append("*** ").append(member.name()); // field names are unique so ignore descriptor
    }
    return result;
  }

  /**
   * Transform a list of Path to a list of InputFileProvider and register them with the given
   * closer.
   */
  @SuppressWarnings("MustBeClosedChecker")
  private static ImmutableList<InputFileProvider> toRegisteredInputFileProvider(
      Closer closer, List<Path> paths) throws IOException {
    ImmutableList.Builder<InputFileProvider> builder = new ImmutableList.Builder<>();
    for (Path path : paths) {
      builder.add(closer.register(InputFileProvider.open(path)));
    }
    return builder.build();
  }

  private KeepScanner() {}
}
