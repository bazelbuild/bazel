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

import com.google.common.collect.Sets;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.bazel.repository.MavenConnector;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;

import org.apache.maven.model.Dependency;
import org.apache.maven.model.Exclusion;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.artifact.DefaultArtifact;

import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Objects;
import java.util.Set;

/**
 * A struct representing the fields of maven_jar to be written to the WORKSPACE file.
 */
public final class Rule implements Comparable<Rule> {
  private final Artifact artifact;
  private final Set<String> parents;
  private String repository;
  private String sha1;
  private Set<String> exclusions;
  private Set<Rule> dependencies;

  public Rule(String artifactStr) throws InvalidRuleException {
    try {
      this.artifact = new DefaultArtifact(artifactStr);
    } catch (IllegalArgumentException e) {
      throw new InvalidRuleException(e.getMessage());
    }
    this.parents = Sets.newHashSet();
    this.dependencies = Sets.newTreeSet();
    this.exclusions = Sets.newHashSet();
  }

  public Rule(Dependency dependency) throws InvalidRuleException {
    this(dependency.getGroupId() + ":" + dependency.getArtifactId() + ":"
        + dependency.getVersion());

    for (Exclusion exclusion : dependency.getExclusions()) {
      exclusions.add(String.format("%s:%s", exclusion.getGroupId(), exclusion.getArtifactId()));
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
      String jarSha1Url = url.replaceAll("pom$", "jar.sha1");
      try {
        // Download the sha1 of the jar file from the repository.
        HttpURLConnection connection = (HttpURLConnection) new URL(jarSha1Url).openConnection();
        connection.setInstanceFollowRedirects(true);
        connection.connect();
        this.sha1 = CharStreams.toString(new InputStreamReader(connection.getInputStream()));
      } catch (IOException e) {
        handler.handle(Event.warn("Failed to download the sha1 at " + jarSha1Url));
      }
    }
  }

  private String getUri() {
    return groupId().replaceAll("\\.", "/") + "/" + artifactId() + "/" + version() + "/"
        + artifactId() + "-" + version() + ".pom";
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
    return repository != null && !repository.equals(MavenConnector.getMavenCentral().getUrl());
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

  /**
   * Exception thrown if the rule could not be created.
   */
  public static class InvalidRuleException extends Exception {
    InvalidRuleException(String message) {
      super(message);
    }
  }
}
