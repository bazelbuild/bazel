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

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import org.apache.maven.model.Model;
import org.apache.maven.model.Repository;
import org.apache.maven.model.building.DefaultModelBuilderFactory;
import org.apache.maven.model.building.DefaultModelBuildingRequest;
import org.apache.maven.model.building.DefaultModelProcessor;
import org.apache.maven.model.building.ModelBuildingException;
import org.apache.maven.model.building.ModelBuildingResult;
import org.apache.maven.model.io.DefaultModelReader;
import org.apache.maven.model.locator.DefaultModelLocator;
import org.eclipse.aether.collection.CollectRequest;
import org.eclipse.aether.repository.RemoteRepository;

import java.io.File;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Resolves Maven dependencies.
 */
public class Resolver {
  private final File projectDirectory;
  private final Map<String, Rule> deps;
  private final EventHandler handler;

  public Resolver(File projectDirectory, EventHandler handler) {
    this.projectDirectory = projectDirectory;
    deps = new HashMap<String, Rule>();
    this.handler = handler;
  }

  /**
   * Find the pom.xml, parse it, and write the equivalent WORKSPACE file to the provided
   * outputStream.
   */
  public void writeDependencies(PrintStream outputStream) {
    resolveDependencies();
    for (Rule rule : deps.values()) {
      outputStream.println(rule.toString() + "\n");
    }
  }

  private void resolveDependencies() {
    DefaultModelProcessor processor = new DefaultModelProcessor();
    processor.setModelLocator(new DefaultModelLocator());
    processor.setModelReader(new DefaultModelReader());
    File pom = processor.locatePom(projectDirectory);

    DefaultModelBuilderFactory factory = new DefaultModelBuilderFactory();
    DefaultModelBuildingRequest request = new DefaultModelBuildingRequest();
    request.setPomFile(pom);
    Model model = null;
    try {
      ModelBuildingResult result = factory.newInstance().build(request);
      model = result.getEffectiveModel();
    } catch (ModelBuildingException e) {
      System.err.println("Unable to resolve Maven model from " + pom + ": " + e.getMessage());
      return;
    }

    CollectRequest collectRequest = new CollectRequest();
    for (Repository repo : model.getRepositories()) {
      collectRequest.addRepository(
          new RemoteRepository.Builder(repo.getId(), repo.getName(), repo.getUrl()).build());
    }

    for (org.apache.maven.model.Dependency dependency : model.getDependencies()) {
      Rule rule = new Rule(
          dependency.getArtifactId(), dependency.getGroupId(), dependency.getVersion());
      if (deps.containsKey(rule.name())) {
        Rule existingDependency = deps.get(rule.name());
        // Check that the versions are the same.
        if (!existingDependency.version().equals(dependency.getVersion())) {
          handler.handle(new Event(EventKind.ERROR, null, dependency.getGroupId() + ":"
              + dependency.getArtifactId() + " already processed for version "
              + existingDependency.version() + " but " + model + " wants version "
              + dependency.getVersion() + ", ignoring."));
        }
        // If it already exists at the right version, we're done.
      } else {
        deps.put(rule.name(), rule);
        // TODO(kchodorow): fetch transitive dependencies.
      }
    }
  }
}
