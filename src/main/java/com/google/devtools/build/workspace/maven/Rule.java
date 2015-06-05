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

import com.google.auto.value.AutoValue;

/**
 * A struct representing the fields of maven_jar to be written to the WORKSPACE file.
 */
@AutoValue
public abstract class Rule {
  static Rule create(String artifactId, String groupId, String version) {
    return new AutoValue_Rule(artifactId, groupId, version);
  }

  abstract String artifactId();
  abstract String groupId();
  abstract String version();

  String name() {
    return (groupId() + "/" + artifactId()).replaceAll("\\.", "/");
  }

  @Override
  public String toString() {
    return "maven_jar(\n"
        + "    name = \"" + name() + "\",\n"
        + "    artifact_id = \"" + artifactId() + "\",\n"
        + "    group_id = \"" + groupId() + "\",\n"
        + "    version = \"" + version() + "\",\n"
        + ")";
  }
}
