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

import com.google.common.collect.Sets;

import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.artifact.DefaultArtifact;

import java.util.Set;

/**
 * A struct representing the fields of maven_jar to be written to the WORKSPACE file.
 */
public final class Rule {
  private final Artifact artifact;
  private final Set<String> parents;
  private String repository;

  public Rule(String artifactStr) throws InvalidRuleException {
    try {
      this.artifact = new DefaultArtifact(artifactStr);
    } catch (IllegalArgumentException e) {
      throw new InvalidRuleException(e.getMessage());
    }
    this.parents = Sets.newHashSet();
  }

  public Rule(String artifactId, String groupId, String version)
      throws InvalidRuleException {
    this(groupId + ":" + artifactId + ":" + version);
  }

  public void addParent(String parent) {
    parents.add(parent);
  }

  public String artifactId() {
    return artifact.getArtifactId();
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
    return (groupId + "/" + artifactId).replaceAll("\\.", "/");
  }

  public Artifact getArtifact() {
    return artifact;
  }

  public String toMavenArtifactString() {
    return groupId() + ":" + artifactId() + ":" + version();
  }

  public void setRepository(String repository) {
    this.repository = repository;
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
        + (repository == null ? "" : "    repository = \"" + repository + "\",\n")
        + ")");
    return builder.toString();
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
