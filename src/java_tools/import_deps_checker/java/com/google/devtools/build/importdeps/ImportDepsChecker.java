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

import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.jar.Attributes;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
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
  private final ImmutableSet<Path> inputJars;

  public ImportDepsChecker(
      ImmutableSet<Path> bootclasspath,
      ImmutableSet<Path> directClasspath,
      ImmutableSet<Path> classpath,
      ImmutableSet<Path> inputJars,
      boolean checkMissingMembers)
      throws IOException {
    this.classCache =
        new ClassCache(
            bootclasspath,
            directClasspath,
            classpath,
            inputJars,
            /*populateMembers=*/ checkMissingMembers);
    this.resultCollector = new ResultCollector(checkMissingMembers);
    this.inputJars = inputJars;
  }

  /**
   * Checks for dependency problems in the given input jars against the classpath.
   *
   * @return {@literal true} for no problems, {@literal false} otherwise.
   */
  public boolean check() throws IOException {
    for (Path path : inputJars) {
      try (ZipFile jarFile = new ZipFile(path.toFile())) {
        jarFile.stream()
            .forEach(
                entry -> {
                  String name = entry.getName();
                  if (!name.endsWith(".class") || name.startsWith("META-INF/versions/")) {
                    // Ignore META-INF/versions/ since given bootclasspath may not cover them, and
                    // any classes would usually only differ in using newer language features.
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
    ImmutableMap<Path, Boolean> paths = classCache.collectUsedJarsInRegularClasspath();
    paths.forEach(
        (path, explicit) ->
            builder.addDependency(
                Dependency.newBuilder()
                    .setKind(explicit ? Kind.EXPLICIT : Kind.IMPLICIT)
                    .setPath(path.toString())
                    .build()));
    return builder.setRuleLabel(ruleLabel).setSuccess(true).build();
  }

  private static final String INDENT = "    ";

  public String computeResultOutput(String ruleLabel) {
    StringBuilder builder = new StringBuilder();
    ImmutableList<String> missingClasses = resultCollector.getSortedMissingClassInternalNames();
    outputMissingClasses(builder, missingClasses);

    ImmutableList<IncompleteState> incompleteClasses = resultCollector.getSortedIncompleteClasses();
    outputIncompleteClasses(builder, incompleteClasses);

    ImmutableList<MissingMember> missingMembers = resultCollector.getSortedMissingMembers();
    outputMissingMembers(builder, missingMembers);

    outputStatistics(builder, missingClasses, incompleteClasses, missingMembers);

    emitAddDepCommandForIndirectJars(ruleLabel, builder);
    return builder.toString();
  }

  private void emitAddDepCommandForIndirectJars(String ruleLabel, StringBuilder builder) {
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
        builder.append("*** Command to add missing strict dependencies: ***\n\n");
        builder.append("    add_dep ");
        for (String indirectLabel : labels) {
          builder.append(indirectLabel).append(" ");
        }
        builder.append(ruleLabel).append('\n');
      }
    }
  }

  private void outputStatistics(
      StringBuilder builder,
      ImmutableList<String> missingClasses,
      ImmutableList<IncompleteState> incompleteClasses,
      ImmutableList<MissingMember> missingMembers) {
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
  }

  private void outputMissingMembers(
      StringBuilder builder, ImmutableList<MissingMember> missingMembers) {
    LinkedHashSet<ClassInfo> classesWithMissingMembers = new LinkedHashSet<>();
    for (MissingMember missing : missingMembers) {
      builder
          .append("Missing member '")
          .append(missing.memberName())
          .append("' in class ")
          .append(missing.owner().internalName().replace('/', '.'))
          .append(" : name=")
          .append(missing.memberName())
          .append(", descriptor=")
          .append(missing.descriptor())
          .append('\n');
      classesWithMissingMembers.add(missing.owner());
    }
    if (!classesWithMissingMembers.isEmpty()) {
      builder.append("The class hierarchies of the classes with missing members:").append("\n");
      classesWithMissingMembers.forEach(
          missingClass -> printClassHierarchy(missingClass, builder, "    "));
    }
  }

  private static void printClassHierarchy(
      ClassInfo klass,
      StringBuilder builder,
      String indent) {
    builder.append(indent).append(toLabeledClassName(klass)).append('\n');
    String superIndent = indent + "    ";

    for (ClassInfo superClass : klass.superClasses()) {
      printClassHierarchy(superClass, builder, superIndent);
    }
  }

  private void outputIncompleteClasses(
      StringBuilder builder, ImmutableList<IncompleteState> incompleteClasses) {
    new LinkedHashMap<>();
    HashMultimap<String, ClassInfo> map = HashMultimap.create();
    for (IncompleteState incomplete : incompleteClasses) {
      ResolutionFailureChain chain = incomplete.resolutionFailureChain();
      map.putAll(chain.getMissingClassesWithSubclasses());
    }
    map.asMap().entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .forEach(
            entry -> {
              builder
                  .append("Indirectly missing class ")
                  .append(entry.getKey().replace('/', '.'))
                  .append(". Referenced by:")
                  .append('\n');
              entry.getValue().stream()
                  .distinct()
                  .sorted()
                  .forEach(
                      reference -> {
                        builder.append(INDENT).append(toLabeledClassName(reference)).append('\n');
                      });
            });
  }

  private void outputMissingClasses(StringBuilder builder, ImmutableList<String> missingClasses) {
    for (String missing : missingClasses) {
      builder.append("Missing ").append(missing.replace('/', '.')).append('\n');
    }
  }

  private static final String toLabeledClassName(ClassInfo klass) {
    String klassName = klass.internalName().replace('/', '.');
    String targetName = extractLabel(klass.jarPath());
    if (targetName != null) {
      int index = targetName.lastIndexOf('/');
      if (index >= 0) {
        // Just print the target name without the full path, as the Bazel tests have
        // different full paths of targets.
        targetName = targetName.substring(index + 1);
      }
      return klassName + " (in " + targetName + ")";
    } else {
      return klassName;
    }
  }

  private static ImmutableList<String> extractLabels(ImmutableList<Path> jars) {
    return jars.stream()
        .map(ImportDepsChecker::extractLabel)
        .filter(Objects::nonNull)
        .distinct()
        .sorted()
        .collect(ImmutableList.toImmutableList());
  }

  private static final Attributes.Name TARGET_LABEL = new Attributes.Name("Target-Label");
  private static final Attributes.Name INJECTING_RULE_KIND =
      new Attributes.Name("Injecting-Rule-Kind");

  @Nullable
  private static String extractLabel(Path jarPath) {
    try (JarFile jar = new JarFile(jarPath.toFile())) {
      Manifest manifest = jar.getManifest();
      if (manifest == null) {
        return null;
      }
      Attributes attributes = manifest.getMainAttributes();
      if (attributes == null) {
        return null;
      }
      String targetLabel = (String) attributes.get(TARGET_LABEL);
      String injectingRuleKind = (String) attributes.get(INJECTING_RULE_KIND);
      if (injectingRuleKind == null) {
        return targetLabel;
      } else {
        return String.format("\"%s %s\"", targetLabel, injectingRuleKind);
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  @Override
  public void close() throws IOException {
    classCache.close();
  }
}
