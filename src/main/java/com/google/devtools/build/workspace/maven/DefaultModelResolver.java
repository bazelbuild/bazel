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
import com.google.devtools.build.lib.bazel.repository.MavenConnector;
import org.apache.maven.model.Parent;
import org.apache.maven.model.Repository;
import org.apache.maven.model.building.ModelSource;
import org.apache.maven.model.building.UrlModelSource;
import org.apache.maven.model.resolution.InvalidRepositoryException;
import org.apache.maven.model.resolution.ModelResolver;
import org.apache.maven.model.resolution.UnresolvableModelException;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

class DefaultModelResolver implements ModelResolver {

  private final List<Repository> repositories;

  public DefaultModelResolver() {
    repositories = Lists.newArrayList();
  }

  private DefaultModelResolver(List<Repository> repositories) {
    this.repositories = repositories;
  }

  @Override
  public ModelSource resolveModel(String groupId, String artifactId, String version)
      throws UnresolvableModelException {
    for (Repository repository : repositories) {
      UrlModelSource modelSource = getModelSource(
          repository.getUrl(), groupId, artifactId, version);
      if (modelSource != null) {
        return modelSource;
      }
    }
    UrlModelSource modelSource = getModelSource(
      MavenConnector.getMavenCentral().getUrl(), groupId, artifactId, version);
    if (modelSource == null) {
      throw new UnresolvableModelException("Could not find any repositories that knew how to "
          + "resolve the artifact (checked " + Arrays.toString(repositories.toArray()) + ")",
          groupId, artifactId, version);
    }
    return modelSource;
  }

  private UrlModelSource getModelSource(
      String url, String groupId, String artifactId, String version)
      throws UnresolvableModelException {
    try {
      UrlModelSource urlModelSource = new UrlModelSource(new URL(url
          + groupId.replaceAll("\\.", "/") + "/" + artifactId + "/" + version + "/" + artifactId
          + "-" + version + ".pom"));
      if (urlModelSource.getInputStream().available() != 0) {
        return urlModelSource;
      }
    } catch (MalformedURLException e) {
      throw new UnresolvableModelException(e.getMessage(), groupId, artifactId, version, e);
    } catch (IOException e) {
      // The artifact could not be fetched from the current repo, just move on and check the next
      // one.
    }
    return null;
  }

  @Override
  public ModelSource resolveModel(Parent parent) throws UnresolvableModelException {
    return resolveModel(parent.getGroupId(), parent.getArtifactId(), parent.getVersion());
  }

  @Override
  public void addRepository(Repository repository) throws InvalidRepositoryException {
    repositories.add(repository);
  }

  @Override
  public void addRepository(Repository repository, boolean replace)
      throws InvalidRepositoryException {
    addRepository(repository);
  }

  @Override
  public ModelResolver newCopy() {
    return new DefaultModelResolver(repositories);
  }
}