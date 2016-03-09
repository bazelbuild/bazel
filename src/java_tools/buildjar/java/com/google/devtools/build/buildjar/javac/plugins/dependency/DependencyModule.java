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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.base.Verify;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.devtools.build.lib.view.proto.Deps.Dependency.Kind;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Wrapper class for managing dependencies on top of
 * {@link com.google.devtools.build.buildjar.javac.BlazeJavaCompiler}. If strict_java_deps is
 * enabled, it keeps two maps between jar names (as they appear on the classpath) and their
 * originating targets, one for direct dependencies and the other for transitive (indirect)
 * dependencies, and enables the {@link StrictJavaDepsPlugin} to perform the actual checks. The
 * plugin also collects dependency information during compilation, and DependencyModule generates a
 * .jdeps artifact summarizing the discovered dependencies.
 */
public final class DependencyModule {

  public static enum StrictJavaDeps {
    /** Legacy behavior: Silently allow referencing transitive dependencies. */
    OFF(false),
    /** Warn about transitive dependencies being used directly. */
    WARN(true),
    /** Fail the build when transitive dependencies are used directly. */
    ERROR(true);

    private final boolean enabled;

    StrictJavaDeps(boolean enabled) {
      this.enabled = enabled;
    }

    /** Convenience method for just checking if it's not OFF */
    public boolean isEnabled() {
      return enabled;
    }
  }

  private final StrictJavaDeps strictJavaDeps;
  private final Map<String, String> directJarsToTargets;
  private final Map<String, String> indirectJarsToTargets;
  private final boolean strictClasspathMode;
  private final Set<String> depsArtifacts;
  private final String ruleKind;
  private final String targetLabel;
  private final String outputDepsFile;
  private final String outputDepsProtoFile;
  private final Set<String> usedClasspath;
  private final Map<String, Deps.Dependency> explicitDependenciesMap;
  private final Map<String, Deps.Dependency> implicitDependenciesMap;
  Set<String> requiredClasspath;
  private final String fixMessage;
  private final Set<String> exemptGenerators;

  DependencyModule(StrictJavaDeps strictJavaDeps,
                   Map<String, String> directJarsToTargets,
                   Map<String, String> indirectJarsToTargets,
                   boolean strictClasspathMode,
                   Set<String> depsArtifacts,
                   String ruleKind,
                   String targetLabel,
                   String outputDepsFile,
                   String outputDepsProtoFile,
                   String fixMessage,
                   Set<String> exemptGenerators) {
    this.strictJavaDeps = strictJavaDeps;
    this.directJarsToTargets = directJarsToTargets;
    this.indirectJarsToTargets = indirectJarsToTargets;
    this.strictClasspathMode = strictClasspathMode;
    this.depsArtifacts = depsArtifacts;
    this.ruleKind = ruleKind;
    this.targetLabel = targetLabel;
    this.outputDepsFile = outputDepsFile;
    this.outputDepsProtoFile = outputDepsProtoFile;
    this.explicitDependenciesMap = new HashMap<>();
    this.implicitDependenciesMap = new HashMap<>();
    this.usedClasspath = new HashSet<>();
    this.fixMessage = fixMessage;
    this.exemptGenerators = exemptGenerators;
  }

  /**
   * Returns a plugin to be enabled in the compiler.
   */
  public BlazeJavaCompilerPlugin getPlugin() {
    return new StrictJavaDepsPlugin(this);
  }

  /**
   * Writes the true, used compile-time classpath to the deps file, if specified.
   */
  public void emitUsedClasspath(String classpath) throws IOException {
    if (outputDepsFile != null) {
      try (BufferedWriter out = new BufferedWriter(new FileWriter(outputDepsFile))) {
        // Filter using the original classpath, to preserve ordering.
        for (String entry : classpath.split(":")) {
          if (usedClasspath.contains(entry)) {
            out.write(entry);
            out.newLine();
          }
        }
      } catch (IOException ex) {
        throw new IOException("Cannot write dependencies to " + outputDepsFile, ex);
      }
    }
  }

  /**
   * Writes dependency information to the deps file in proto format, if specified.
   *
   * This is a replacement for {@link #emitUsedClasspath} above, which only outputs the used
   * classpath. We collect more precise dependency information to allow Blaze to analyze both
   * strict and unused dependencies based on the new deps.proto.
   */
  public void emitDependencyInformation(String classpath, boolean successful) throws IOException {
    if (outputDepsProtoFile == null) {
      return;
    }

    try (BufferedOutputStream out =
             new BufferedOutputStream(new FileOutputStream(outputDepsProtoFile))) {
      buildDependenciesProto(classpath, successful).writeTo(out);
    } catch (IOException ex) {
      throw new IOException("Cannot write dependencies to " + outputDepsProtoFile, ex);
    }
  }

  @VisibleForTesting
  Deps.Dependencies buildDependenciesProto(String classpath, boolean successful) {
    Deps.Dependencies.Builder deps = Deps.Dependencies.newBuilder();
    if (targetLabel != null) {
      deps.setRuleLabel(targetLabel);
    }
    deps.setSuccess(successful);
    // Filter using the original classpath, to preserve ordering.
    for (String entry : classpath.split(":")) {
      if (explicitDependenciesMap.containsKey(entry)) {
        deps.addDependency(explicitDependenciesMap.get(entry));
      } else if (implicitDependenciesMap.containsKey(entry)) {
        deps.addDependency(implicitDependenciesMap.get(entry));
      }
    }
    return deps.build();
  }

  /**
   * Returns whether strict dependency checks (strictJavaDeps) are enabled.
   */
  public boolean isStrictDepsEnabled() {
    return strictJavaDeps.isEnabled();
  }

  /**
   * Returns the mapping for jars of direct dependencies. The keys are full
   * paths (as seen on the classpath), and the values are build target names.
   */
  public Map<String, String> getDirectMapping() {
    return directJarsToTargets;
  }

  /**
   * Returns the mapping for jars of indirect dependencies. The keys are full
   * paths (as seen on the classpath), and the values are build target names.
   */
  public Map<String, String> getIndirectMapping() {
    return indirectJarsToTargets;
  }

  /**
   * Returns the strict dependency checking (strictJavaDeps) setting.
   */
  public StrictJavaDeps getStrictJavaDeps() {
    return strictJavaDeps;
  }

  /**
   * Returns the map collecting precise explicit dependency information.
   */
  public Map<String, Deps.Dependency> getExplicitDependenciesMap() {
    return explicitDependenciesMap;
  }

  /**
   * Returns the map collecting precise implicit dependency information.
   */
  public Map<String, Deps.Dependency> getImplicitDependenciesMap() {
    return implicitDependenciesMap;
  }

  /**
   * Returns the type (rule kind) of the originating target.
   */
  public String getRuleKind() {
    return ruleKind;
  }

  /**
   * Returns the name (label) of the originating target.
   */
  public String getTargetLabel() {
    return targetLabel;
  }

  /**
   * Returns the file name collecting dependency information.
   */
  public String getOutputDepsFile() {
    return outputDepsFile;
  }

  @VisibleForTesting
  Set<String> getUsedClasspath() {
    return usedClasspath;
  }

  /**
   * Returns a message to suggest fix when a missing indirect dependency is found.
   */
  public String getFixMessage() {
    return fixMessage;
  }

  /**
   * Return a set of generator values that are exempt from strict dependencies.
   */
  public Set<String> getExemptGenerators() {
    return exemptGenerators;
  }

  /**
   * Returns whether classpath reduction is enabled for this invocation.
   */
  public boolean reduceClasspath() {
    return strictClasspathMode;
  }

  private static final Splitter CLASSPATH_SPLITTER = Splitter.on(':');
  private static final Joiner CLASSPATH_JOINER = Joiner.on(File.pathSeparator);

  /**
   * Computes a reduced compile-time classpath from the union of direct dependencies and their
   * dependencies, as listed in the associated .deps artifacts.
   */
  public String computeStrictClasspath(String originalClasspath) throws IOException {
    if (!strictClasspathMode) {
      return originalClasspath;
    }

    return CLASSPATH_JOINER.join(
        computeStrictClasspath(CLASSPATH_SPLITTER.split(originalClasspath)));
  }

  /**
   * Computes a reduced compile-time classpath from the union of direct dependencies and their
   * dependencies, as listed in the associated .deps artifacts.
   */
  public List<String> computeStrictClasspath(Iterable<String> originalClasspath)
      throws IOException {
    Verify.verify(strictClasspathMode);

    // Classpath = direct deps + runtime direct deps + their .deps
    requiredClasspath = new HashSet<>(directJarsToTargets.keySet());

    for (String depsArtifact : depsArtifacts) {
       collectDependenciesFromArtifact(depsArtifact);
    }

    // Filter the initial classpath and keep the original order
    List<String> filteredClasspath = new ArrayList<>();
    for (String entry : originalClasspath) {
      if (requiredClasspath.contains(entry)) {
        filteredClasspath.add(entry);
      }
    }
    return filteredClasspath;
  }

  @VisibleForTesting
  void setStrictClasspath(Set<String> strictClasspath) {
    this.requiredClasspath = strictClasspath;
  }

  /**
   * Updates {@link #requiredClasspath} to include dependencies from the given output artifact.
   */
  private void collectDependenciesFromArtifact(String path) throws IOException {
    try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(path))) {
      Deps.Dependencies deps = Deps.Dependencies.parseFrom(bis);
      // Sanity check to make sure we have a valid proto.
      if (!deps.hasRuleLabel()) {
        throw new IOException("Could not parse Deps.Dependencies message from proto.");
      }
      for (Deps.Dependency dep : deps.getDependencyList()) {
        if (dep.getKind() == Kind.EXPLICIT || dep.getKind() == Kind.IMPLICIT) {
          requiredClasspath.add(dep.getPath());
        }
      }
    } catch (IOException e) {
      throw new IOException(String.format("error reading deps artifact: %s", path), e);
    }
  }

  /**
   * Builder for {@link DependencyModule}.
   */
  public static class Builder {

    private StrictJavaDeps strictJavaDeps = StrictJavaDeps.OFF;
    private final Map<String, String> directJarsToTargets = new HashMap<>();
    private final Map<String, String> indirectJarsToTargets = new HashMap<>();
    private final Set<String> depsArtifacts = new HashSet<>();
    private String ruleKind;
    private String targetLabel;
    private String outputDepsFile;
    private String outputDepsProtoFile;
    private boolean strictClasspathMode = false;
    private String fixMessage = "%s** Please add the following dependencies:%s\n"
        + "  %s to %s\n\n";
    private final Set<String> exemptGenerators = new HashSet<>();

    /**
     * Constructs the DependencyModule, guaranteeing that the maps are
     * never null (they may be empty), and the default strictJavaDeps setting
     * is OFF.
     *
     * @return an instance of DependencyModule
     */
    public DependencyModule build() {
      return new DependencyModule(strictJavaDeps, directJarsToTargets, indirectJarsToTargets,
          strictClasspathMode, depsArtifacts, ruleKind, targetLabel, outputDepsFile,
          outputDepsProtoFile, fixMessage, exemptGenerators);
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
     * Sets the type (rule kind) of the originating target.
     *
     * @param ruleKind kind, such as the rule kind of a RuleConfiguredTarget
     * @return this Builder instance
     */
    public Builder setRuleKind(String ruleKind) {
      this.ruleKind = ruleKind;
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

    /**
     * Adds a direct mapping to the existing map for direct dependencies.
     *
     * @param jar path of jar artifact, as seen on classpath.
     * @param target full name of build target providing the jar.
     * @return this Builder instance.
     */
    public Builder addDirectMapping(String jar, String target) {
      directJarsToTargets.put(jar, target);
      return this;
    }

    /**
     * Adds direct mappings to the existing map for direct dependencies.
     *
     * @param directMappings a map of paths of jar artifacts, as seen on classpath, to full names of
     *     build targets providing the jar.
     * @return this Builder instance
     */
    public Builder addDirectMappings(Map<String, String> directMappings) {
      directJarsToTargets.putAll(directMappings);
      return this;
    }

    /**
     * Adds an indirect mapping to the existing map for indirect dependencies.
     *
     * @param jar path of jar artifact, as seen on classpath.
     * @param target full name of build target providing the jar.
     * @return this Builder instance
     */
    public Builder addIndirectMapping(String jar, String target) {
      indirectJarsToTargets.put(jar, target);
      return this;
    }

    /**
     * Adds indirect mappings to the existing map for indirect dependencies.
     *
     * @param indirectMappings a map of paths of jar artifacts, as seen on classpath, to full names
     *     of build targets providing the jar.
     * @return this Builder instance
     */
    public Builder addIndirectMappings(Map<String, String> indirectMappings) {
      indirectJarsToTargets.putAll(indirectMappings);
      return this;
    }

    /**
     * Sets the name of the file that will contain dependency information.
     *
     * @param outputDepsFile output file name for dependency information
     * @return this Builder instance
     */
    public Builder setOutputDepsFile(String outputDepsFile) {
      this.outputDepsFile = outputDepsFile;
      return this;
    }

    /**
     * Sets the name of the file that will contain dependency information in the protocol buffer
     * format.
     *
     * @param outputDepsProtoFile output file name for dependency information
     * @return this Builder instance
     */
    public Builder setOutputDepsProtoFile(String outputDepsProtoFile) {
      this.outputDepsProtoFile = outputDepsProtoFile;
      return this;
    }

    /**
     * Adds a collection of dependency artifacts to use when reducing the compile-time classpath.
     *
     * @param depsArtifacts dependency artifacts
     * @return this Builder instance
     */
    public Builder addDepsArtifacts(Collection<String> depsArtifacts) {
      this.depsArtifacts.addAll(depsArtifacts);
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
    public Builder setFixMessage(String fixMessage) {
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
