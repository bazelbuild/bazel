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
import com.google.common.io.Files;
import com.google.devtools.build.lib.bazel.repository.MavenConnector;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.vfs.FileSystem;

import org.apache.maven.model.Model;
import org.apache.maven.model.Repository;
import org.apache.maven.model.building.DefaultModelBuilderFactory;
import org.apache.maven.model.building.DefaultModelBuildingRequest;
import org.apache.maven.model.building.DefaultModelProcessor;
import org.apache.maven.model.building.ModelBuildingException;
import org.apache.maven.model.building.ModelBuildingResult;
import org.apache.maven.model.io.DefaultModelReader;
import org.apache.maven.model.locator.DefaultModelLocator;
import org.apache.maven.model.resolution.InvalidRepositoryException;
import org.apache.maven.model.resolution.ModelResolver;
import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.RepositorySystemSession;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.graph.Dependency;
import org.eclipse.aether.resolution.ArtifactDescriptorException;
import org.eclipse.aether.resolution.ArtifactDescriptorRequest;
import org.eclipse.aether.resolution.ArtifactDescriptorResult;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;

/**
 * Resolves Maven dependencies.
 */
public class Resolver {
  private final MavenConnector connector;
  private final File localRepository;
  private final FileSystem fileSystem;
  private final EventHandler handler;

  private final List<String> headers;
  // Mapping of maven_jar name to Rule.
  private final Map<String, Rule> deps;

  public Resolver(EventHandler handler, FileSystem fileSystem) {
    this.handler = handler;
    this.fileSystem = fileSystem;
    this.localRepository = Files.createTempDir();
    this.connector = new MavenConnector(localRepository.getPath());
    headers = Lists.newArrayList();
    deps = Maps.newHashMap();
  }

  /**
   * Writes all resolved dependencies in WORKSPACE file format to the outputStream.
   */
  public void writeDependencies(PrintStream outputStream) {
    outputStream.println("# --------------------\n"
        + "# The following dependencies were calculated from:");
    for (String header : headers) {
      outputStream.println("# " + header);
    }
    outputStream.print("\n\n");
    for (Rule rule : deps.values()) {
      outputStream.println(rule + "\n");
    }
    outputStream.println("# --------------------\n");
  }

  public void addHeader(String header) {
    headers.add(header);
  }

  /**
   * Remove the temporary directory storing pom files.
   */
  public void cleanup() {
    try {
      for (File file : Files.fileTreeTraverser().postOrderTraversal(localRepository)) {
        java.nio.file.Files.delete(file.toPath());
      }
    } catch (IOException e) {
      handler.handle(Event.error(Location.fromFile(fileSystem.getPath(localRepository.getPath())),
          "Could not create local repository directory " + localRepository + ": "
              + e.getMessage()));
    }
  }

  /**
   * Resolves all dependencies from a pom.xml file in the given directory.
   */
  public void resolvePomDependencies(String project) {
    DefaultModelProcessor processor = new DefaultModelProcessor();
    processor.setModelLocator(new DefaultModelLocator());
    processor.setModelReader(new DefaultModelReader());
    File pom = processor.locatePom(new File(project));
    Location pomLocation = Location.fromFile(fileSystem.getPath(pom.getPath()));
    addHeader(pom.getAbsolutePath());

    DefaultModelBuilderFactory factory = new DefaultModelBuilderFactory();
    DefaultModelBuildingRequest request = new DefaultModelBuildingRequest();
    ModelResolver modelResolver = new DefaultModelResolver();
    request.setModelResolver(modelResolver);
    request.setPomFile(pom);
    Model model;
    try {
      ModelBuildingResult result = factory.newInstance().build(request);
      model = result.getEffectiveModel();
    } catch (ModelBuildingException | IllegalArgumentException e) {
      // IllegalArg can be thrown if the parent POM cannot be resolved.
      handler.handle(Event.error(pomLocation,
          "Unable to resolve Maven model from " + pom + ": " + e.getMessage()));
      return;
    }

    for (Repository repo : model.getRepositories()) {
      try {
        modelResolver.addRepository(repo);
      } catch (InvalidRepositoryException e) {
        handler.handle(Event.error("Unable to add repository " + repo.getName()
            + " (" + repo.getId() + "," + repo.getUrl() + ")"));
        return;
      }
    }

    for (String module : model.getModules()) {
      resolvePomDependencies(project + "/" + module);
    }

    for (org.apache.maven.model.Dependency dependency : model.getDependencies()) {
      try {
        Rule artifactRule = new Rule(
            dependency.getArtifactId(), dependency.getGroupId(), dependency.getVersion());
        addArtifact(artifactRule, model.toString());
        getArtifactDependencies(artifactRule, pomLocation);
      } catch (Rule.InvalidRuleException e) {
        handler.handle(Event.error(pomLocation, e.getMessage()));
      }
    }
  }

  /**
   * Adds transitive dependencies of the given artifact.
   */
  public void getArtifactDependencies(Rule artifactRule, Location location) {
    Artifact artifact = artifactRule.getArtifact();

    RepositorySystem system = connector.newRepositorySystem();
    RepositorySystemSession session = connector.newRepositorySystemSession(system);

    ArtifactDescriptorRequest descriptorRequest = new ArtifactDescriptorRequest();
    descriptorRequest.setArtifact(artifact);
    descriptorRequest.addRepository(MavenConnector.getMavenCentral());

    ArtifactDescriptorResult descriptorResult;
    try {
      descriptorResult = system.readArtifactDescriptor(session, descriptorRequest);
      if (descriptorResult == null) {
        return;
      }
    } catch (ArtifactDescriptorException e) {
      handler.handle(Event.error(location, e.getMessage()));
      return;
    }

    for (Dependency dependency : descriptorResult.getDependencies()) {
      Artifact depArtifact = dependency.getArtifact();
      try {
        Rule rule = new Rule(
            depArtifact.getArtifactId(), depArtifact.getGroupId(), depArtifact.getVersion());
        if (addArtifact(rule, artifactRule.toMavenArtifactString())) {
          getArtifactDependencies(rule, location);
        }
      } catch (Rule.InvalidRuleException e) {
        handler.handle(Event.error(location, e.getMessage()));
      }
    }
  }

  /**
   * Adds the artifact to the list of deps, if it is not already there. Returns if the artifact
   * was already in the list. If the artifact was in the list at a different version, adds an
   * error event to the event handler.
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
