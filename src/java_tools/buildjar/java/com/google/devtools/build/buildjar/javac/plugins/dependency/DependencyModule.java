// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac.plugins.dependency;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Streams;
import com.google.devtools.build.buildjar.JarOwner;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.lib.view.proto.Deps.Dependencies;
import com.google.devtools.build.lib.view.proto.Deps.Dependency;
import com.google.devtools.build.lib.view.proto.Deps.Dependency.Kind;
import com.sun.tools.javac.code.Symbol.PackageSymbol;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import javax.tools.Diagnostic;
import javax.tools.JavaFileObject;

/**
 * Wrapper class for managing dependencies on top of {@link
 * com.google.devtools.build.buildjar.javac.BlazeJavaCompiler}. If strict_java_deps is enabled, it
 * keeps two maps between jar names (as they appear on the classpath) and their originating targets,
 * one for direct dependencies and the other for transitive (indirect) dependencies, and enables the
 * {@link StrictJavaDepsPlugin} to perform the actual checks. The plugin also collects dependency
 * information during compilation, and DependencyModule generates a .jdeps artifact summarizing the
 * discovered dependencies.
 */
public final class DependencyModule {

  public static enum StrictJavaDeps {
    /** Legacy behavior: Silently allow referencing transitive dependencies. */
    OFF,
    /** Warn about transitive dependencies being used directly. */
    WARN,
    /** Fail the build when transitive dependencies are used directly. */
    ERROR
  }

  private static final ImmutableSet<String> SJD_EXEMPT_PROCESSORS =
      ImmutableSet.of(
          // Relax strict deps for dagger-generated code (b/17979436).
          "dagger.internal.codegen.ComponentProcessor");

  private final StrictJavaDeps strictJavaDeps;
  private final FixTool fixDepsTool;
  private final ImmutableSet<Path> directJars;
  private final boolean strictClasspathMode;
  private final Set<Path> depsArtifacts;
  private final String targetLabel;
  private final Path outputDepsProtoFile;
  private boolean hasMissingTargets;
  private final Map<Path, Dependency> explicitDependenciesMap;
  private final Map<Path, Dependency> implicitDependenciesMap;
  private final ImmutableSet<Path> platformJars;
  Set<Path> requiredClasspath;
  private final FixMessage fixMessage;
  private final Set<String> exemptGenerators;
  private final Set<PackageSymbol> packages;

  DependencyModule(
      StrictJavaDeps strictJavaDeps,
      FixTool fixDepsTool,
      ImmutableSet<Path> directJars,
      boolean strictClasspathMode,
      Set<Path> depsArtifacts,
      ImmutableSet<Path> platformJars,
      String targetLabel,
      Path outputDepsProtoFile,
      FixMessage fixMessage,
      Set<String> exemptGenerators) {
    this.strictJavaDeps = strictJavaDeps;
    this.fixDepsTool = fixDepsTool;
    this.directJars = directJars;
    this.strictClasspathMode = strictClasspathMode;
    this.depsArtifacts = depsArtifacts;
    this.targetLabel = targetLabel;
    this.outputDepsProtoFile = outputDepsProtoFile;
    this.explicitDependenciesMap = new HashMap<>();
    this.implicitDependenciesMap = new HashMap<>();
    this.platformJars = platformJars;
    this.fixMessage = fixMessage;
    this.exemptGenerators = exemptGenerators;
    this.packages = new HashSet<>();
  }

  /** Returns a plugin to be enabled in the compiler. */
  public BlazeJavaCompilerPlugin getPlugin() {
    return new StrictJavaDepsPlugin(this);
  }

  /**
   * Writes dependency information to the deps file in proto format, if specified.
   *
   * <p>We collect precise dependency information to allow Blaze to analyze both strict and unused
   * dependencies, as well as packages contained by the output jar.
   */
  public void emitDependencyInformation(
      ImmutableList<Path> classpath, boolean successful, boolean requiresFallback)
      throws IOException {
    if (outputDepsProtoFile == null) {
      return;
    }

    try (BufferedOutputStream out =
        new BufferedOutputStream(Files.newOutputStream(outputDepsProtoFile))) {
      buildDependenciesProto(classpath, successful, requiresFallback).writeTo(out);
    } catch (IOException ex) {
      throw new IOException("Cannot write dependencies to " + outputDepsProtoFile, ex);
    }
  }

  @VisibleForTesting
  Dependencies buildDependenciesProto(
      ImmutableList<Path> classpath, boolean successful, boolean requiresFallback) {
    Dependencies.Builder deps = Dependencies.newBuilder();
    if (targetLabel != null) {
      deps.setRuleLabel(targetLabel);
    }
    deps.setSuccess(successful);
    if (requiresFallback) {
      deps.setRequiresReducedClasspathFallback(true);
    }

    deps.addAllContainedPackage(
        packages
            .stream()
            .map(pkg -> pkg.isUnnamed() ? "" : pkg.getQualifiedName().toString())
            .sorted()
            .collect(toImmutableList()));

    // Filter using the original classpath, to preserve ordering.
    for (Path entry : classpath) {
      if (explicitDependenciesMap.containsKey(entry)) {
        deps.addDependency(explicitDependenciesMap.get(entry));
      } else if (implicitDependenciesMap.containsKey(entry)) {
        deps.addDependency(implicitDependenciesMap.get(entry));
      }
    }
    return deps.build();
  }

  /** Returns the paths of direct dependencies. */
  public ImmutableSet<Path> directJars() {
    return directJars;
  }

  /** Returns the strict dependency checking (strictJavaDeps) setting. */
  public StrictJavaDeps getStrictJavaDeps() {
    return strictJavaDeps;
  }

  /** Returns which tool to use for adding missing dependencies. */
  public FixTool getFixDepsTool() {
    return fixDepsTool;
  }

  /** Returns the map collecting precise explicit dependency information. */
  public Map<Path, Dependency> getExplicitDependenciesMap() {
    return explicitDependenciesMap;
  }

  /** Returns the map collecting precise implicit dependency information. */
  public Map<Path, Dependency> getImplicitDependenciesMap() {
    return implicitDependenciesMap;
  }

  /** Returns the jars in the platform classpath. */
  public ImmutableSet<Path> getPlatformJars() {
    return platformJars;
  }

  /** Adds a package to the set of packages built by this target. */
  public boolean addPackage(PackageSymbol packge) {
    return packages.add(packge);
  }

  /** Returns the name (label) of the originating target. */
  public String getTargetLabel() {
    return targetLabel;
  }

  /** Returns the file name collecting dependency information. */
  public Path getOutputDepsProtoFile() {
    return outputDepsProtoFile;
  }

  /** Returns a message to suggest fix when a missing indirect dependency is found. */
  public FixMessage getFixMessage() {
    return fixMessage;
  }

  /** Return a set of generator values that are exempt from strict dependencies. */
  public Set<String> getExemptGenerators() {
    return exemptGenerators;
  }

  /** Returns whether classpath reduction is enabled for this invocation. */
  public boolean reduceClasspath() {
    return strictClasspathMode;
  }

  void setHasMissingTargets() {
    hasMissingTargets = true;
  }

  /** Returns true if any missing transitive dependencies were reported. */
  public boolean hasMissingTargets() {
    return hasMissingTargets;
  }

  /**
   * Computes a reduced compile-time classpath from the union of direct dependencies and their
   * dependencies, as listed in the associated .deps artifacts.
   */
  public ImmutableList<Path> computeStrictClasspath(ImmutableList<Path> originalClasspath)
      throws IOException {
    if (!strictClasspathMode) {
      return originalClasspath;
    }

    // Classpath = direct deps + runtime direct deps + their .deps
    requiredClasspath = new HashSet<>(directJars);

    for (Path depsArtifact : depsArtifacts) {
      collectDependenciesFromArtifact(depsArtifact);
    }

    // TODO(b/71936047): it should be an error for requiredClasspath to contain paths that are not
    // in originalClasspath

    // Filter the initial classpath and keep the original order
    return originalClasspath
        .stream()
        .filter(requiredClasspath::contains)
        .collect(toImmutableList());
  }

  @VisibleForTesting
  void setStrictClasspath(Set<Path> strictClasspath) {
    this.requiredClasspath = strictClasspath;
  }

  /** Updates {@link #requiredClasspath} to include dependencies from the given output artifact. */
  private void collectDependenciesFromArtifact(Path path) throws IOException {
    try (BufferedInputStream bis = new BufferedInputStream(Files.newInputStream(path))) {
      Dependencies deps = Dependencies.parseFrom(bis);
      // Sanity check to make sure we have a valid proto.
      if (!deps.hasRuleLabel()) {
        throw new IOException("Could not parse Deps.Dependencies message from proto.");
      }
      for (Dependency dep : deps.getDependencyList()) {
        if (dep.getKind() == Kind.EXPLICIT
            || dep.getKind() == Kind.IMPLICIT
            || dep.getKind() == Kind.INCOMPLETE) {
          requiredClasspath.add(Paths.get(dep.getPath()));
        }
      }
    } catch (IOException e) {
      throw new IOException(String.format("error reading deps artifact: %s", path), e);
    }
  }

  /** Emits a message to the user about missing dependencies to add to unbreak their build. */
  public interface FixMessage {

    /**
     * Gets a message describing what dependencies are missing and how to fix them.
     *
     * @param missing the missing dependencies to be added.
     * @param recipient the target from which the dependencies are missing.
     * @return the string message describing the dependency build issues, including fix.
     */
    String get(Iterable<JarOwner> missing, String recipient);
  }

  /** Tool with which to fix dependency issues. */
  public interface FixTool {

    /**
     * Applies this tool to find the missing import/dependency.
     *
     * @param diagnostic a full javac diagnostic, possibly containing an import for a class which
     *     cannot be found on the classpath.
     * @param javacopts list of all javac options/flags.
     * @return the missing import or dependency as a String, or empty Optional if the diagnostic did
     *     not contain exactly one unresolved import that we know how to fix.
     */
    Optional<String> resolveMissingImport(
        Diagnostic<JavaFileObject> diagnostic, ImmutableList<String> javacopts);

    /**
     * Returns a command for this tool to fix {@code recipient} by adding all {@code missing}
     * dependencies for this target.
     */
    String getFixCommand(Iterable<String> missing, String recipient);
  }

  /** Builder for {@link DependencyModule}. */
  public static class Builder {

    private StrictJavaDeps strictJavaDeps = StrictJavaDeps.OFF;
    private FixTool fixDepsTool = null;
    private ImmutableSet<Path> directJars = ImmutableSet.of();
    private final Set<Path> depsArtifacts = new HashSet<>();
    private ImmutableSet<Path> platformJars = ImmutableSet.of();
    private String targetLabel;
    private Path outputDepsProtoFile;
    private boolean strictClasspathMode = false;
    private FixMessage fixMessage = new DefaultFixMessage();
    private final Set<String> exemptGenerators = new LinkedHashSet<>(SJD_EXEMPT_PROCESSORS);

    private static class DefaultFixMessage implements FixMessage {
      @Override
      public String get(Iterable<JarOwner> missing, String recipient) {
        ImmutableSet<String> missingTargets =
            Streams.stream(missing)
                .flatMap(owner -> owner.label().map(Stream::of).orElse(Stream.empty()))
                .collect(toImmutableSet());
        if (missingTargets.isEmpty()) {
          return "";
        }
        return String.format(
            "%1$s ** Please add the following dependencies:%2$s \n  %3$s to %4$s \n"
                + "%1$s ** You can use the following buildozer command:%2$s "
                + "\nbuildozer 'add deps %3$s' %4$s \n\n",
            "\033[35m\033[1m", "\033[0m", Joiner.on(" ").join(missingTargets), recipient);
      }
    }

    /**
     * Constructs the DependencyModule, guaranteeing that the maps are never null (they may be
     * empty), and the default strictJavaDeps setting is OFF.
     *
     * @return an instance of DependencyModule
     */
    public DependencyModule build() {
      return new DependencyModule(
          strictJavaDeps,
          fixDepsTool,
          directJars,
          strictClasspathMode,
          depsArtifacts,
          platformJars,
          targetLabel,
          outputDepsProtoFile,
          fixMessage,
          exemptGenerators);
    }

    /**
     * Sets the strictness level for dependency checking.
     *
     * @param strictJavaDeps level, as specified by {@link StrictJavaDeps}
     * @return this Builder instance
     */
    public Builder setStrictJavaDeps(String strictJavaDeps) {
      this.strictJavaDeps = StrictJavaDeps.valueOf(strictJavaDeps);
      return this;
    }

    /**
     * Sets which tool to use for fixing missing dependencies.
     *
     * @param fixDepsTool tool name
     * @return this Builder instance
     */
    public Builder setFixDepsTool(FixTool fixDepsTool) {
      this.fixDepsTool = fixDepsTool;
      return this;
    }

    /**
     * Sets the name (label) of the originating target.
     *
     * @param targetLabel label, such as the label of a RuleConfiguredTarget.
     * @return this Builder instance.
     */
    public Builder setTargetLabel(String targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }

    /** Sets the paths to jars that are direct dependencies. */
    public Builder setDirectJars(ImmutableSet<Path> directJars) {
      this.directJars = directJars;
      return this;
    }

    /**
     * Sets the name of the file that will contain dependency information in the protocol buffer
     * format.
     *
     * @param outputDepsProtoFile output file name for dependency information
     * @return this Builder instance
     */
    public Builder setOutputDepsProtoFile(Path outputDepsProtoFile) {
      this.outputDepsProtoFile = outputDepsProtoFile;
      return this;
    }

    /**
     * Adds a collection of dependency artifacts to use when reducing the compile-time classpath.
     *
     * @param depsArtifacts dependency artifacts
     * @return this Builder instance
     */
    public Builder addDepsArtifacts(Collection<Path> depsArtifacts) {
      this.depsArtifacts.addAll(depsArtifacts);
      return this;
    }

    /** Sets the platform classpath entries. */
    public Builder setPlatformJars(ImmutableSet<Path> platformJars) {
      this.platformJars = platformJars;
      return this;
    }

    /**
     * Requests compile-time classpath reduction based on provided dependency artifacts.
     *
     * @return this Builder instance
     */
    public Builder setReduceClasspath() {
      this.strictClasspathMode = true;
      return this;
    }

    /**
     * Set the message to display when a missing indirect dependency is found.
     *
     * @param fixMessage the fix message
     * @return this Builder instance
     */
    public Builder setFixMessage(FixMessage fixMessage) {
      this.fixMessage = fixMessage;
      return this;
    }

    /**
     * Add a generator to the exempt set.
     *
     * @param exemptGenerator the generator class name
     * @return this Builder instance
     */
    public Builder addExemptGenerator(String exemptGenerator) {
      exemptGenerators.add(exemptGenerator);
      return this;
    }
  }
}
