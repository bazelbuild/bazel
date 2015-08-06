// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

import org.apache.maven.model.Model;
import org.apache.maven.model.Repository;
import org.apache.maven.model.building.DefaultModelBuilder;
import org.apache.maven.model.building.DefaultModelBuilderFactory;
import org.apache.maven.model.building.DefaultModelBuildingRequest;
import org.apache.maven.model.building.DefaultModelProcessor;
import org.apache.maven.model.building.FileModelSource;
import org.apache.maven.model.building.ModelBuildingException;
import org.apache.maven.model.building.ModelBuildingResult;
import org.apache.maven.model.building.ModelSource;
import org.apache.maven.model.io.DefaultModelReader;
import org.apache.maven.model.locator.DefaultModelLocator;
import org.apache.maven.model.resolution.InvalidRepositoryException;
import org.apache.maven.model.resolution.UnresolvableModelException;

import java.io.File;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;

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

  public Resolver(EventHandler handler) {
    this.handler = handler;
    this.headers = Lists.newArrayList();
    this.deps = Maps.newHashMap();
    this.modelBuilder = new DefaultModelBuilderFactory().newInstance();
    this.modelResolver = new DefaultModelResolver();
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
    outputStream.println("java_library(");
    outputStream.println("    name = \"transitive-deps\",");
    outputStream.println("    visibility = [\"//visibility:public\"],");
    outputStream.println("    exports = [");
    for (Rule rule : deps.values()) {
      outputStream.println("        \"@" + rule.name() + "//jar\",");
    }
    outputStream.println("    ],");
    outputStream.println(")");
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
    Model model = resolveModelSource(new FileModelSource(pom));

    // For the top-level pom _only_, resolve all of its submodules.
    resolveSubmodules(model, pom);
  }

  /**
   * This calls resolvePomDependencies on each submodule of the model, thus filling in the
   * transitive submodules' dependencies as well as the main project's.
   */
  private void resolveSubmodules(Model model, File pom) {
    for (String module : model.getModules()) {
      resolvePomDependencies(pom.getParent() + "/" + module);
    }
  }

  /**
   * Resolves all dependencies from a given "model source," which could be either a URL or a local
   * file.
   * @return the model.
   */
  @Nullable
  public Model resolveModelSource(ModelSource modelSource) {
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
      try {
        modelResolver.addRepository(repo);
      } catch (InvalidRepositoryException e) {
        handler.handle(Event.error("Unable to add repository " + repo.getName()
            + " (" + repo.getId() + "," + repo.getUrl() + ")"));
        return model;
      }
    }

    for (org.apache.maven.model.Dependency dependency : model.getDependencies()) {
      if (!dependency.getScope().equals(COMPILE_SCOPE)) {
        continue;
      }
      try {
        Rule artifactRule = new Rule(dependency);
        boolean isNewDependency = addArtifact(artifactRule, model.toString());
        if (isNewDependency) {
          ModelSource depModelSource = modelResolver.resolveModel(
              dependency.getGroupId(), dependency.getArtifactId(), dependency.getVersion());
          if (depModelSource != null) {
            artifactRule.setRepository(depModelSource.getLocation());
            resolveModelSource(depModelSource);
          } else {
            handler.handle(Event.error("Could not get a model for " + dependency));
          }
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
}
