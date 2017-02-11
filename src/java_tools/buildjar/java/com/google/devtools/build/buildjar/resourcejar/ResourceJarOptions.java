// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.resourcejar;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;

/** Resource jar builder options. */
public class ResourceJarOptions {
  private final String output;
  private final ImmutableList<String> messages;
  private final ImmutableList<String> resources;
  private final ImmutableList<String> resourceJars;
  private final ImmutableList<String> classpathResources;

  public ResourceJarOptions(
      String output,
      ImmutableList<String> messages,
      ImmutableList<String> resources,
      ImmutableList<String> resourceJars,
      ImmutableList<String> classpathResources) {
    this.output = output;
    this.messages = messages;
    this.resources = resources;
    this.resourceJars = resourceJars;
    this.classpathResources = classpathResources;
  }

  public String output() {
    return output;
  }

  /**
   * Resources to include in the jar.
   *
   * <p>The format is {@code <prefix>:<name>}, where {@code <prefix>/<name>} is the path to the
   * resource file, and {code <name>} is the relative name that will be used for the resource jar
   * entry.
   */
  public ImmutableList<String> resources() {
    return resources;
  }

  /** Message files to include in the resource jar. The format is the same as {@link #resources}. */
  public ImmutableList<String> messages() {
    return messages;
  }

  /** Jar files of resources to append to the resource jar. */
  public ImmutableList<String> resourceJars() {
    return resourceJars;
  }

  /** Files to include as top-level entries in the resource jar. */
  public ImmutableList<String> classpathResources() {
    return classpathResources;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A builder for {@link ResourceJarOptions}. */
  public static class Builder {
    private String output;
    private ImmutableList<String> messages = ImmutableList.of();
    private ImmutableList<String> resources = ImmutableList.of();
    private ImmutableList<String> resourceJars = ImmutableList.of();
    private ImmutableList<String> classpathResources = ImmutableList.of();

    public ResourceJarOptions build() {
      return new ResourceJarOptions(output, messages, resources, resourceJars, classpathResources);
    }

    public Builder setOutput(String output) {
      this.output = checkNotNull(output);
      return this;
    }

    public Builder setMessages(ImmutableList<String> messages) {
      this.messages = checkNotNull(messages);
      return this;
    }

    public Builder setResources(ImmutableList<String> resources) {
      this.resources = checkNotNull(resources);
      return this;
    }

    public Builder setResourceJars(ImmutableList<String> resourceJars) {
      this.resourceJars = checkNotNull(resourceJars);
      return this;
    }

    public Builder setClasspathResources(ImmutableList<String> classpathResources) {
      this.classpathResources = checkNotNull(classpathResources);
      return this;
    }
  }
}
