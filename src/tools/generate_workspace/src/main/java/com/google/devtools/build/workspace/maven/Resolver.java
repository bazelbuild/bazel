// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.workspace.maven;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

import org.apache.maven.model.Model;
import org.apache.maven.model.Parent;
import org.apache.maven.model.Repository;
import org.apache.maven.model.building.DefaultModelBuilder;
import org.apache.maven.model.building.DefaultModelBuilderFactory;
import org.apache.maven.model.building.DefaultModelBuildingRequest;
import org.apache.maven.model.building.DefaultModelProcessor;
import org.apache.maven.model.building.FileModelSource;
import org.apache.maven.model.building.ModelBuildingException;
import org.apache.maven.model.building.ModelBuildingResult;
import org.apache.maven.model.building.ModelSource;
import org.apache.maven.model.composition.DefaultDependencyManagementImporter;
import org.apache.maven.model.io.DefaultModelReader;
import org.apache.maven.model.locator.DefaultModelLocator;
import org.apache.maven.model.management.DefaultDependencyManagementInjector;
import org.apache.maven.model.management.DefaultPluginManagementInjector;
import org.apache.maven.model.plugin.DefaultPluginConfigurationExpander;
import org.apache.maven.model.profile.DefaultProfileSelector;
import org.apache.maven.model.resolution.UnresolvableModelException;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Resolves Maven dependencies.
 */
public class Resolver {
  private static final String COMPILE_SCOPE = "compile";

  private final EventHandler handler;
  private final DefaultModelBuilder modelBuilder;
  private final DefaultModelResolver modelResolver;

  private final List<String> headers;
  // Mapping of maven_jar name to Rule.
  private final Map<String, Rule> deps;
  private Set<Rule> rootDependencies;

  public Resolver(EventHandler handler) {
    this.handler = handler;
    this.headers = Lists.newArrayList();
    this.deps = Maps.newHashMap();
    this.modelBuilder = new DefaultModelBuilderFactory().newInstance()
        .setProfileSelector(new DefaultProfileSelector())
        .setPluginConfigurationExpander(new DefaultPluginConfigurationExpander())
        .setPluginManagementInjector(new DefaultPluginManagementInjector())
        .setDependencyManagementImporter(new DefaultDependencyManagementImporter())
        .setDependencyManagementInjector(new DefaultDependencyManagementInjector());
    this.modelResolver = new DefaultModelResolver();
    this.rootDependencies = Sets.newTreeSet();
  }

  /**
   * Writes all resolved dependencies in WORKSPACE file format to the outputStream.
   */
  public void writeWorkspace(PrintStream outputStream) {
    writeHeader(outputStream);
    for (Rule rule : deps.values()) {
      outputStream.println(rule + "\n");
    }
  }

  /**
   * Write library rules to depend on the transitive closure of all of these rules.
   */
  public void writeBuild(PrintStream outputStream) {
    writeHeader(outputStream);
    for (Rule rule : rootDependencies) {
      outputStream.println("java_library(");
      outputStream.println("    name = \"" + rule.name() + "\",");
      outputStream.println("    visibility = [\"//visibility:public\"],");
      outputStream.println("    exports = [");
      outputStream.println("        \"@" + rule.name() + "//jar\",");
      for (Rule r : rule.getDependencies()) {
        outputStream.println("        \"@" + r.name() + "//jar\",");
      }
      outputStream.println("    ],");
      outputStream.println(")");
    }
  }

  private void writeHeader(PrintStream outputStream) {
    outputStream.println("# The following dependencies were calculated from:");
    for (String header : headers) {
      outputStream.println("# " + header);
    }
    outputStream.print("\n\n");
  }

  public void addHeader(String header) {
    headers.add(header);
  }

  public DefaultModelResolver getModelResolver() {
    return modelResolver;
  }

  /**
   * Given a local path to a Maven project, this attempts to find the transitive dependencies of
   * the project.
   * @param projectPath The path to search for Maven projects.
   */
  public void resolvePomDependencies(String projectPath) {
    DefaultModelProcessor processor = new DefaultModelProcessor();
    processor.setModelLocator(new DefaultModelLocator());
    processor.setModelReader(new DefaultModelReader());
    File pom = processor.locatePom(new File(projectPath));
    addHeader(pom.getAbsolutePath());
    FileModelSource pomSource = new FileModelSource(pom);
    // First resolve the model source locations.
    resolveSourceLocations(pomSource);
    // Next, fully resolve the models.
    resolveEffectiveModel(pomSource, Sets.<String>newHashSet(), null);
  }

  /**
   * Resolves all dependencies from a given "model source," which could be either a URL or a local
   * file.
   * @return the model.
   */
  @Nullable
  public Model resolveEffectiveModel(ModelSource modelSource, Set<String> exclusions, Rule parent) {
    DefaultModelBuildingRequest request = new DefaultModelBuildingRequest();
    request.setModelResolver(modelResolver);
    request.setModelSource(modelSource);
    Model model;
    try {
      ModelBuildingResult result = modelBuilder.build(request);
      model = result.getEffectiveModel();
    } catch (ModelBuildingException | IllegalArgumentException e) {
      // IllegalArg can be thrown if the parent POM cannot be resolved.
      handler.handle(Event.error("Unable to resolve Maven model from " + modelSource.getLocation()
          + ": " + e.getMessage()));
      return null;
    }
    for (Repository repo : model.getRepositories()) {
      modelResolver.addRepository(repo);
    }

    for (org.apache.maven.model.Dependency dependency : model.getDependencies()) {
      if (!dependency.getScope().equals(COMPILE_SCOPE)) {
        continue;
      }
      if (dependency.isOptional()) {
        continue;
      }
      if (exclusions.contains(dependency.getGroupId() + ":" + dependency.getArtifactId())) {
        continue;
      }
      try {
        Rule artifactRule = new Rule(dependency);
        HashSet<String> localDepExclusions = new HashSet<>(exclusions);
        localDepExclusions.addAll(artifactRule.getExclusions());

        boolean isNewDependency = addArtifact(artifactRule, model.toString());
        if (isNewDependency) {
          ModelSource depModelSource = modelResolver.resolveModel(
              dependency.getGroupId(), dependency.getArtifactId(), dependency.getVersion());
          if (depModelSource != null) {
            artifactRule.setRepository(depModelSource.getLocation(), handler);
            artifactRule.setSha1(downloadSha1(artifactRule));
            resolveEffectiveModel(depModelSource, localDepExclusions, artifactRule);
          } else {
            handler.handle(Event.error("Could not get a model for " + dependency));
          }
        }

        if (parent != null) {
          parent.addDependency(artifactRule);
          parent.getDependencies().addAll(artifactRule.getDependencies());
        } else {
          rootDependencies.add(artifactRule);
        }
      } catch (UnresolvableModelException | Rule.InvalidRuleException e) {
        handler.handle(Event.error("Could not resolve dependency " + dependency.getGroupId()
            + ":" + dependency.getArtifactId() + ":" + dependency.getVersion() + ": "
            + e.getMessage()));
      }
    }
    return model;
  }

  /**
   * Find the POM files for a given pom's parent(s) and submodules.
   */
  private void resolveSourceLocations(FileModelSource fileModelSource) {
    DefaultModelBuildingRequest request = new DefaultModelBuildingRequest();
    request.setModelResolver(modelResolver);
    request.setModelSource(fileModelSource);
    Model model;
    try {
      ModelBuildingResult result = modelBuilder.build(request);
      model = result.getRawModel();
    } catch (ModelBuildingException | IllegalArgumentException e) {
      // IllegalArg can be thrown if the parent POM cannot be resolved.
      handler.handle(Event.error("Unable to resolve raw Maven model from "
          + fileModelSource.getLocation() + ": " + e.getMessage()));
      return;
    }

    // Self.
    Parent parent = model.getParent();
    if (model.getGroupId() == null) {
      model.setGroupId(parent.getGroupId());
    }
    if (!modelResolver.putModelSource(
        model.getGroupId(), model.getArtifactId(), fileModelSource)) {
      return;
    }

    // Parent.
    File pomDirectory = new File(fileModelSource.getLocation()).getParentFile();
    if (parent != null && !parent.getArtifactId().equals(model.getArtifactId())) {
      File parentPom;
      try {
        parentPom = new File(pomDirectory, parent.getRelativePath()).getCanonicalFile();
      } catch (IOException e) {
        handler.handle(Event.error("Unable to get canonical path of " + pomDirectory + " and "
            + parent.getRelativePath()));
        return;
      }
      if (parentPom.exists()) {
        resolveSourceLocations(new FileModelSource(parentPom));
      }
    }

    // Submodules.
    for (String module : model.getModules()) {
      resolveSourceLocations(new FileModelSource(new File(pomDirectory, module + "/pom.xml")));
    }
  }

  /**
   * Adds the artifact to the map of deps, if it is not already there. Returns if the artifact
   * was newly added. If the artifact was in the list at a different version, adds an annotation
   * about the desired version.
   */
  private boolean addArtifact(Rule dependency, String parent) {
    String artifactName = dependency.name();
    if (deps.containsKey(artifactName)) {
      Rule existingDependency = deps.get(artifactName);
      // Check that the versions are the same.
      if (!existingDependency.version().equals(dependency.version())) {
        existingDependency.addParent(parent + " wanted version " + dependency.version());
      } else {
        existingDependency.addParent(parent);
      }
      return false;
    }

    deps.put(artifactName, dependency);
    dependency.addParent(parent);
    return true;
  }

  public void addRootDependency(Rule rule) {
    rootDependencies.add(rule);
  }

  static String getSha1Url(String url, String extension) {
    return url.replaceAll(".pom$", "." + extension + ".sha1");
  }

  /**
   * Downloads the SHA-1 for the given artifact.
   */
  public String downloadSha1(Rule rule) {
    String sha1Url = getSha1Url(rule.getUrl(), rule.getArtifact().getExtension());
    try {
      HttpURLConnection connection = (HttpURLConnection) new URL(sha1Url).openConnection();
      connection.setInstanceFollowRedirects(true);
      connection.connect();
      return CharStreams.toString(
          new InputStreamReader(connection.getInputStream(), Charset.defaultCharset())).trim();
    } catch (IOException e) {
      handler.handle(Event.warn("Failed to download the sha1 at " + sha1Url));
    }
    return null;
  }
}
