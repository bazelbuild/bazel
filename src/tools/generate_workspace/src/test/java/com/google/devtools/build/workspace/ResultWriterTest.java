// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.workspace;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.workspace.maven.Rule;

import org.eclipse.aether.artifact.DefaultArtifact;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Set;

/**
 * Tests for @{link ResultWriter}.
 */
@RunWith(JUnit4.class)
public class ResultWriterTest {

  public String getWorkspaceFileContent(List<String> sources, Set<Rule> rules) throws Exception{
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos);
    ResultWriter writer = new ResultWriter(sources, rules);
    writer.writeWorkspace(ps);
    return baos.toString(String.valueOf(Charset.defaultCharset()));
  }

  public String getBuildFileContent(List<String> sources, Set<Rule> rules) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos);
    ResultWriter writer = new ResultWriter(sources, rules);
    writer.writeBuild(ps);
    return baos.toString(String.valueOf(Charset.defaultCharset()));
  }

  @Test
  public void testHeaders() throws Exception {
    String content = getWorkspaceFileContent(
        ImmutableList.of("foo", "bar"), Sets.<Rule>newHashSet());
    assertThat(content).contains("# foo\n# bar");
  }

  @Test
  public void testArtifacts() throws Exception {
    Set<Rule> rules = ImmutableSet.of(
        new Rule(new DefaultArtifact("x:y:1.2.3")));
    String content = getWorkspaceFileContent(ImmutableList.<String>of(), rules);
    assertThat(content).contains("maven_jar(\n"
        + "    name = \"x_y\",\n"
        + "    artifact = \"x:y:1.2.3\",\n"
        + ")"
    );
  }

  @Test
  public void testParents() throws Exception {
    Rule rule = new Rule(new DefaultArtifact("x:y:1.2.3"));
    rule.addParent("some parent");
    Set<Rule> rules = ImmutableSet.of(rule);
    String content = getWorkspaceFileContent(ImmutableList.<String>of(), rules);
    assertThat(content).contains("# some parent\n"
            + "maven_jar(\n"
            + "    name = \"x_y\",\n"
            + "    artifact = \"x:y:1.2.3\",\n"
            + ")"
    );
  }

  @Test
  public void testBuildFile() throws Exception {
    Rule rule = new Rule(new DefaultArtifact("x:y:1.2.3"));
    Rule dep1 = new Rule(new DefaultArtifact("dep:dep1:4.5.6"));
    rule.addDependency(dep1);
    Rule dep2 = new Rule(new DefaultArtifact("dep:dep2:7.8.9"));
    rule.addDependency(dep2);
    Set<Rule> rules = ImmutableSet.of(rule, dep1, dep2);
    String content = getBuildFileContent(ImmutableList.<String>of(), rules);
    assertThat(content).contains("java_library(\n"
            + "    name = \"x_y\",\n"
            + "    visibility = [\"//visibility:public\"],\n"
            + "    exports = [\n"
            + "        \"@x_y//jar\",\n"
            + "        \"@dep_dep1//jar\",\n"
            + "        \"@dep_dep2//jar\",\n"
            + "    ],\n"
            + ")"
    );
  }
}
