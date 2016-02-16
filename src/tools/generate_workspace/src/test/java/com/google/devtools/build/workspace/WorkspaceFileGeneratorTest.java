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

package com.google.devtools.build.workspace;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.workspace.maven.Resolver;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.File;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Tests for generating a WORKSPACE file.
 */
@RunWith(JUnit4.class)
public class WorkspaceFileGeneratorTest {

  @Test
  public void testResolver() throws Exception {
    File tmpdir = new File(System.getenv("TEST_TMPDIR"));
    String pom = tmpdir + "/pom.xml";
    PrintWriter pomWriter = new PrintWriter(pom);
    pomWriter.println(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<project xmlns=\"http://maven.apache.org/POM/4.0.0\""
            + "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
            + "xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 "
            + "http://maven.apache.org/xsd/maven-4.0.0.xsd\">"
            + "  <modelVersion>4.0.0</modelVersion>"
            + "  <version>1.0</version>"
            + "  <groupId>com.google.appengine.demos</groupId>"
            + "  <artifactId>appengine-try-java</artifactId>"
            + "  <properties>"
            + "    <appengine.target.version>1.9.20</appengine.target.version>"
            + "  </properties>"
            + "  <dependencies>"
            + "    <dependency>"
            + "      <groupId>com.google.appengine</groupId>"
            + "      <artifactId>appengine-api-1.0-sdk</artifactId>"
            + "      <version>${appengine.target.version}</version>"
            + "    </dependency>"
            + " </dependencies>"
            + "</project>");
    pomWriter.close();

    StoredEventHandler handler = new StoredEventHandler();
    Resolver resolver = new Resolver(handler);
    String outputFile = tmpdir + "/output";
    PrintStream outputStream = new PrintStream(outputFile);
    resolver.resolvePomDependencies(tmpdir.getAbsolutePath());
    resolver.writeWorkspace(outputStream);
    outputStream.close();

    assertEquals(
        "# The following dependencies were calculated from:\n"
            + "# "
            + pom
            + "\n\n\n"
            + "# com.google.appengine.demos:appengine-try-java:jar:1.0\n"
            + "maven_jar(\n"
            + "    name = \"com_google_appengine_appengine_api_1_0_sdk\",\n"
            + "    artifact = \"com.google.appengine:appengine-api-1.0-sdk:1.9.20\",\n"
            + ")\n\n",
        new String(Files.readAllBytes(Paths.get(outputFile))));
    // We can't recursively fetch deps over the network.
    assertTrue(handler.hasErrors());
  }
}
