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
import static java.nio.file.StandardOpenOption.CREATE;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.android.Converters.ExistingPathConverter;
import com.google.devtools.build.android.Converters.PathConverter;
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
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
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
      help = "Input Jar with classes to scan."
    )
    public Path inputJars;

    @Option(
      name = "keep_file",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = OptionEffectTag.UNKNOWN,
      converter = PathConverter.class,
      help = "Where to write keep rules to."
    )
    public Path keepDest;

    @Option(
      name = "prefix",
      defaultValue = "j$/",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = OptionEffectTag.UNKNOWN,
      help = "type to scan for."
    )
    public String prefix;
  }

  public static void main(String... args) throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(KeepScannerOptions.class);
    parser.setAllowResidue(false);
    parser.enableParamsFileSupport(new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    parser.parseAndExitUponError(args);

    KeepScannerOptions options = parser.getOptions(KeepScannerOptions.class);
    Map<String, ImmutableSet<KeepReference>> seeds =
        scan(checkNotNull(options.inputJars), options.prefix);

    try (PrintStream out =
        new PrintStream(
            Files.newOutputStream(options.keepDest, CREATE), /*autoFlush=*/ false, "UTF-8")) {
      writeKeepDirectives(out, seeds);
    }
  }

  /**
   * Writes a -keep rule for each class listing any members to keep.  We sort classes and members
   * so the output is deterministic.
   */
  private static void writeKeepDirectives(
      PrintStream out, Map<String, ImmutableSet<KeepReference>> seeds) {
    seeds
        .entrySet()
        .stream()
        .sorted(comparing(Map.Entry::getKey))
        .forEachOrdered(
            type -> {
              out.printf("-keep class %s {%n", type.getKey().replace('/', '.'));
              type.getValue()
                  .stream()
                  .filter(KeepReference::isMemberReference)
                  .sorted(comparing(KeepReference::name).thenComparing(KeepReference::desc))
                  .map(ref -> toKeepDescriptor(ref))
                  .distinct() // drop duplicates due to method descriptors with different returns
                  .forEachOrdered(line -> out.append("  ").append(line).append(";").println());
              out.printf("}%n");
            });
  }

  /**
   * Scans for and returns references with owners matching the given prefix grouped by owner.
   */
  private static Map<String, ImmutableSet<KeepReference>> scan(Path jarFile, String prefix)
      throws IOException {
    // We read the Jar sequentially since ZipFile uses locks anyway but then allow scanning each
    // class in parallel.
    try (ZipFile zip = new ZipFile(jarFile.toFile())) {
      return zip.stream()
          .filter(entry -> entry.getName().endsWith(".class"))
          .map(entry -> readFully(zip, entry))
          .parallel()
          .flatMap(
              content -> PrefixReferenceScanner.scan(new ClassReader(content), prefix).stream())
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

  private KeepScanner() {}
}
