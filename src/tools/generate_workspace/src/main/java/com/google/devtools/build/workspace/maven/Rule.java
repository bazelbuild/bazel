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

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.bazel.repository.MavenConnector;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

import org.apache.maven.model.Exclusion;
import org.eclipse.aether.artifact.Artifact;

import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * A struct representing the fields of maven_jar to be written to the WORKSPACE file.
 */
public final class Rule implements Comparable<Rule> {

  private final Artifact artifact;
  private final Set<String> parents;
  private final Set<String> exclusions;
  private final Set<Rule> dependencies;
  private String repository;
  private String sha1;

  public Rule(Artifact artifact) {
    this.artifact = artifact;
    this.parents = Sets.newHashSet();
    this.dependencies = Sets.newTreeSet();
    this.exclusions = Sets.newHashSet();
    this.repository = MavenConnector.MAVEN_CENTRAL_URL;
  }

  public Rule(Artifact artifact, List<Exclusion> exclusions) {
    this(artifact);

    for (Exclusion exclusion : exclusions) {
      String coord = String.format("%s:%s", exclusion.getGroupId(), exclusion.getArtifactId());
      this.exclusions.add(coord);
    }
  }

  public void addParent(String parent) {
    parents.add(parent);
  }

  public void addDependency(Rule dependency) {
    dependencies.add(dependency);
  }

  public Set<Rule> getDependencies() {
    return dependencies;
  }

  public String artifactId() {
    return artifact.getArtifactId();
  }

  public Set<String> getExclusions() {
    return exclusions;
  }

  public String groupId() {
    return artifact.getGroupId();
  }

  public String version() {
    return artifact.getVersion();
  }

  /**
   * A unique name for this artifact to use in maven_jar's name attribute.
   */
  String name() {
    return Rule.name(groupId(), artifactId());
  }

  /**
   * A unique name for this artifact to use in maven_jar's name attribute.
   */
  public static String name(String groupId, String artifactId) {
    return groupId.replaceAll("[.-]", "_") + "_" + artifactId.replaceAll("[.-]", "_");
  }

  public Artifact getArtifact() {
    return artifact;
  }

  public String toMavenArtifactString() {
    return groupId() + ":" + artifactId() + ":" + version();
  }

  public void setRepository(String url, EventHandler handler) {
    // url is of the form repository/group/artifact/version/artifact-version.pom. Strip off
    // everything after repository/.
    int uriStart = url.indexOf(getUri());
    if (uriStart == -1) {
      // If url is actually a path to a file, it won't match the URL pattern described above.
      // However, in that case we also have no way of fetching the artifact, so we'll print a
      // warning.
      handler.handle(Event.warn(name() + " was defined in " + url
          + " which isn't a repository URL, so we couldn't figure out how to fetch "
          + toMavenArtifactString() + " in a general way. You will need to set the \"repository\""
          + " attribute manually"));
    } else {
      this.repository = url.substring(0, uriStart);
    }
  }

  public void setSha1(String sha1) {
    this.sha1 = sha1;
  }

  private String getUri() {
    return groupId().replaceAll("\\.", "/") + "/" + artifactId() + "/" + version() + "/"
        + artifactId() + "-" + version() + ".pom";
  }

  /**
   * @return The artifact's URL.
   */
  public String getUrl() {
    Preconditions.checkState(repository.endsWith("/"));
    return repository + getUri();
  }

  /**
   * The way this jar should be stringified for the WORKSPACE file.
   */
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (String parent : parents) {
      builder.append("# " + parent + "\n");
    }
    builder.append("maven_jar(\n"
        + "    name = \"" + name() + "\",\n"
        + "    artifact = \"" + toMavenArtifactString() + "\",\n"
        + (hasCustomRepository() ? "    repository = \"" + repository + "\",\n" : "")
        + (sha1 != null ? "    sha1 = \"" + sha1 + "\",\n" : "")
        + ")");
    return builder.toString();
  }

  private boolean hasCustomRepository() {
    return !MavenConnector.MAVEN_CENTRAL_URL.equals(repository);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    Rule rule = (Rule) o;

    return Objects.equals(groupId(), rule.groupId())
        && Objects.equals(artifactId(), rule.artifactId());
  }

  @Override
  public int hashCode() {
    return Objects.hash(groupId(), artifactId());
  }

  @Override
  public int compareTo(Rule o) {
    return name().compareTo(o.name());
  }
}
