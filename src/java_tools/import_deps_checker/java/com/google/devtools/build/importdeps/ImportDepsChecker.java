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
package com.google.devtools.build.importdeps;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.importdeps.AbstractClassEntryState.IncompleteState;
import com.google.devtools.build.importdeps.ClassInfo.MemberInfo;
import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Collectors;
import java.util.zip.ZipFile;
import org.objectweb.asm.ClassReader;

/**
 * Checker that checks the classes in the input jars have complete dependencies. If not, output the
 * missing dependencies to a file.
 */
public class ImportDepsChecker implements Closeable {

  private final ClassCache classCache;
  private final ResultCollector resultCollector;
  private final ImmutableList<Path> inputJars;

  public ImportDepsChecker(
      ImmutableList<Path> bootclasspath,
      ImmutableList<Path> classpath,
      ImmutableList<Path> inputJars)
      throws IOException {
    this.classCache =
        new ClassCache(
            ImmutableList.<Path>builder()
                .addAll(bootclasspath)
                .addAll(classpath)
                .addAll(inputJars)
                .build());
    this.resultCollector = new ResultCollector();
    this.inputJars = inputJars;
  }

  public ImportDepsChecker check() throws IOException {
    for (Path path : inputJars) {
      try (ZipFile jarFile = new ZipFile(path.toFile())) {
        jarFile
            .stream()
            .forEach(
                entry -> {
                  String name = entry.getName();
                  if (!name.endsWith(".class")) {
                    return;
                  }
                  try (InputStream inputStream = jarFile.getInputStream(entry)) {
                    ClassReader reader = new ClassReader(inputStream);
                    DepsCheckerClassVisitor checker =
                        new DepsCheckerClassVisitor(classCache, resultCollector);
                    reader.accept(checker, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES);
                  } catch (IOException e) {
                    throw new IOError(e);
                  }
                });
      }
    }
    return this;
  }

  private static final String INDENT = "    ";

  public void saveResult(Path resultFile) throws IOException {
    if (!Files.exists(resultFile)) {
      Files.createFile(resultFile); // Make sure the file exists.
    }
    try (BufferedWriter writer = Files.newBufferedWriter(resultFile, StandardCharsets.UTF_8)) {
      ImmutableList<String> missingClasses = resultCollector.getSortedMissingClassInternalNames();
      for (String missing : missingClasses) {
        writer.append("Missing ").append(missing.replace('/', '.')).append('\n');
      }

      ImmutableList<IncompleteState> incompleteClasses =
          resultCollector.getSortedIncompleteClasses();
      for (IncompleteState incomplete : incompleteClasses) {
        writer
            .append("Incomplete ancestor classpath for ")
            .append(incomplete.classInfo().get().internalName().replace('/', '.'))
            .append('\n');

        ImmutableList<String> failurePath = incomplete.getResolutionFailurePath();
        checkState(!failurePath.isEmpty(), "The resolution failure path is empty. %s", failurePath);
        writer
            .append(INDENT)
            .append("missing ancestor: ")
            .append(failurePath.get(failurePath.size() - 1).replace('/', '.'))
            .append('\n');
        writer
            .append(INDENT)
            .append("resolution failure path: ")
            .append(
                failurePath
                    .stream()
                    .map(internalName -> internalName.replace('/', '.'))
                    .collect(Collectors.joining(" -> ")))
            .append('\n');
      }
      ImmutableList<MemberInfo> missingMembers = resultCollector.getSortedMissingMembers();
      for (MemberInfo missing : missingMembers) {
        writer
            .append("Missing member '")
            .append(missing.memberName())
            .append("' in class ")
            .append(missing.owner().replace('/', '.'))
            .append(" : name=")
            .append(missing.memberName())
            .append(", descriptor=")
            .append(missing.descriptor())
            .append('\n');
      }
      if (missingClasses.size() + incompleteClasses.size() + missingMembers.size() != 0) {
        writer
            .append("===Total===\n")
            .append("missing=")
            .append(String.valueOf(missingClasses.size()))
            .append('\n')
            .append("incomplete=")
            .append(String.valueOf(incompleteClasses.size()))
            .append('\n')
            .append("missing_members=")
            .append(String.valueOf(missingMembers.size()));
      }
    }
  }

  @Override
  public void close() throws IOException {
    classCache.close();
  }
}
