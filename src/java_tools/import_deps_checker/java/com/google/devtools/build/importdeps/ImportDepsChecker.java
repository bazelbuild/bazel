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
import com.google.devtools.build.importdeps.ResultCollector.MissingMember;
import com.google.devtools.build.lib.view.proto.Deps.Dependencies;
import com.google.devtools.build.lib.view.proto.Deps.Dependency;
import com.google.devtools.build.lib.view.proto.Deps.Dependency.Kind;
import java.io.Closeable;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.jar.JarFile;
import java.util.stream.Collectors;
import java.util.zip.ZipFile;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassReader;

/**
 * Checker that checks the classes in the input jars have complete dependencies. If not, output the
 * missing dependencies to a file.
 */
public final class ImportDepsChecker implements Closeable {

  private final ClassCache classCache;
  private final ResultCollector resultCollector;
  private final ImmutableList<Path> inputJars;

  public ImportDepsChecker(
      ImmutableList<Path> bootclasspath,
      ImmutableList<Path> directClasspath,
      ImmutableList<Path> classpath,
      ImmutableList<Path> inputJars)
      throws IOException {
    this.classCache = new ClassCache(bootclasspath, directClasspath, classpath, inputJars);
    this.resultCollector = new ResultCollector();
    this.inputJars = inputJars;
  }

  /**
   * Checks for dependency problems in the given input jars agains the classpath.
   *
   * @return {@literal true} for no problems, {@literal false} otherwise.
   */
  public boolean check() throws IOException {
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
                  } catch (RuntimeException e) {
                    System.err.printf(
                        "A runtime exception occurred when processing the class %s "
                            + "in the zip file %s\n",
                        name, path);
                    throw e;
                  }
                });
      }
    }
    return resultCollector.isEmpty();
  }

  /** Emit the jdeps proto. The parameter ruleLabel is optional, indicated with the empty string. */
  public Dependencies emitJdepsProto(String ruleLabel) {
    Dependencies.Builder builder = Dependencies.newBuilder();
    ImmutableList<Path> paths = classCache.collectUsedJarsInRegularClasspath();
    // TODO(b/77723273): Consider "implicit" for Jars only needed to resolve supertypes
    paths.forEach(
        path ->
            builder.addDependency(
                Dependency.newBuilder().setKind(Kind.EXPLICIT).setPath(path.toString()).build()));
    return builder.setRuleLabel(ruleLabel).setSuccess(true).build();
  }

  private static final String INDENT = "    ";

  public String computeResultOutput(String ruleLabel) {
    StringBuilder builder = new StringBuilder();
    ImmutableList<String> missingClasses = resultCollector.getSortedMissingClassInternalNames();
    for (String missing : missingClasses) {
      builder.append("Missing ").append(missing.replace('/', '.')).append('\n');
    }

    ImmutableList<IncompleteState> incompleteClasses = resultCollector.getSortedIncompleteClasses();
    for (IncompleteState incomplete : incompleteClasses) {
      builder
          .append("Incomplete ancestor classpath for ")
          .append(incomplete.classInfo().get().internalName().replace('/', '.'))
          .append('\n');

      ImmutableList<String> failurePath = incomplete.getResolutionFailurePath();
      checkState(!failurePath.isEmpty(), "The resolution failure path is empty. %s", failurePath);
      builder
          .append(INDENT)
          .append("missing ancestor: ")
          .append(failurePath.get(failurePath.size() - 1).replace('/', '.'))
          .append('\n');
      builder
          .append(INDENT)
          .append("resolution failure path: ")
          .append(
              failurePath
                  .stream()
                  .map(internalName -> internalName.replace('/', '.'))
                  .collect(Collectors.joining(" -> ")))
          .append('\n');
    }
    ImmutableList<MissingMember> missingMembers = resultCollector.getSortedMissingMembers();
    for (MissingMember missing : missingMembers) {
      builder
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
      builder
          .append("===Total===\n")
          .append("missing=")
          .append(missingClasses.size())
          .append('\n')
          .append("incomplete=")
          .append(incompleteClasses.size())
          .append('\n')
          .append("missing_members=")
          .append(missingMembers.size())
          .append('\n');
    }

    ImmutableList<Path> indirectJars = resultCollector.getSortedIndirectDeps();
    if (!indirectJars.isEmpty()) {
      ImmutableList<String> labels = extractLabels(indirectJars);
      if (ruleLabel.isEmpty() || labels.isEmpty()) {
        builder
            .append(
                "*** Missing strict dependencies on the following Jars which don't carry "
                    + "rule labels.\nPlease determine the originating rules, e.g., using Bazel's "
                    + "'query' command, and add them to the dependencies of ")
            .append(ruleLabel.isEmpty() ? inputJars : ruleLabel)
            .append('\n');
        for (Path jar : indirectJars) {
          builder.append(jar).append('\n');
        }
      } else {
        builder.append("*** Missing strict dependencies. Run the following command to fix ***\n\n");
        builder.append("    add_dep ");
        for (String indirectLabel : labels) {
          builder.append(indirectLabel).append(" ");
        }
        builder.append(ruleLabel).append('\n');
      }
    }
    return builder.toString();
  }

  private static ImmutableList<String> extractLabels(ImmutableList<Path> jars) {
    return jars.parallelStream()
        .map(ImportDepsChecker::extractLabel)
        .filter(Objects::nonNull)
        .distinct()
        .sorted()
        .collect(ImmutableList.toImmutableList());
  }

  @Nullable
  private static String extractLabel(Path jarPath) {
    try (JarFile jar = new JarFile(jarPath.toFile())) {
      return jar.getManifest().getMainAttributes().getValue("Target-Label");
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  @Override
  public void close() throws IOException {
    classCache.close();
  }
}
