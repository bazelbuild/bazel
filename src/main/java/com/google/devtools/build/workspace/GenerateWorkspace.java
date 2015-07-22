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

package com.google.devtools.build.workspace;

import com.google.common.io.Files;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.ExternalPackage;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixFileSystem;
import com.google.devtools.common.options.OptionsParser;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.List;

/**
 * Generates a WORKSPACE file for Bazel from other types of dependency trackers.
 */
public class GenerateWorkspace {

  private final StoredEventHandler handler;
  private final FileSystem fileSystem;
  private final com.google.devtools.build.workspace.maven.Resolver resolver;

  public static void main(String[] args) {
    OptionsParser parser = OptionsParser.newOptionsParser(GenerateWorkspaceOptions.class);
    parser.parseAndExitUponError(args);
    GenerateWorkspaceOptions options = parser.getOptions(GenerateWorkspaceOptions.class);
    if (options.mavenProjects.isEmpty() && options.bazelProjects.isEmpty()) {
      printUsage(parser);
      return;
    }

    GenerateWorkspace workspaceFileGenerator = new GenerateWorkspace();
    workspaceFileGenerator.generateFromWorkspace(options.bazelProjects);
    workspaceFileGenerator.generateFromPom(options.mavenProjects);
    if (!workspaceFileGenerator.hasErrors()) {
      workspaceFileGenerator.writeResults();
    }
    workspaceFileGenerator.cleanup();
    if (workspaceFileGenerator.hasErrors()) {
      System.exit(1);
    }
  }

  private static void printUsage(OptionsParser parser) {
    System.out.println("Usage: generate_workspace (-b PATH|-m PATH)+\n\n"
        + "Generates a workspace file from the given projects. At least one bazel_project or "
        + "maven_project must be specified.\n");
    System.out.println(parser.describeOptions(Collections.<String, String>emptyMap(),
        OptionsParser.HelpVerbosity.LONG));
  }

  private GenerateWorkspace() {
    this.handler = new StoredEventHandler();
    this.fileSystem = getFileSystem();
    this.resolver = new com.google.devtools.build.workspace.maven.Resolver(handler, fileSystem);
  }

  static FileSystem getFileSystem() {
    return OS.getCurrent() == OS.WINDOWS
        ? new JavaIoFileSystem() : new UnixFileSystem();
  }

  private void generateFromWorkspace(List<String> projects) {
    for (String project : projects) {
      Resolver workspaceResolver = new Resolver(resolver, handler);
      Path projectPath = fileSystem.getPath(getAbsolute(project));
      ExternalPackage externalPackage =
          workspaceResolver.parse(projectPath.getRelative("WORKSPACE"));
      workspaceResolver.resolveTransitiveDependencies(externalPackage);
    }
  }

  private void generateFromPom(List<String> projects) {
    for (String project : projects) {
      resolver.resolvePomDependencies(getAbsolute(project));
    }
  }

  private String getAbsolute(String path) {
    return Paths.get(System.getProperty("user.dir")).resolve(path).toString();
  }

  /**
   * Returns if there were any errors generating the WORKSPACE and BUILD files.
   */
  private boolean hasErrors() {
    return handler.hasErrors();
  }

  private void writeResults() {
    String date = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date());
    File tempDir = Files.createTempDir();
    File workspaceFile = new File(tempDir + "/" + date + ".WORKSPACE");
    File buildFile = new File(tempDir + "/" + date + ".BUILD");
    try (PrintStream workspaceStream = new PrintStream(workspaceFile);
         PrintStream buildStream = new PrintStream(buildFile)) {
      resolver.writeWorkspace(workspaceStream);
      resolver.writeBuild(buildStream);
    } catch (IOException e) {
      handler.handle(Event.error(
          "Could not write WORKSPACE and BUILD files to " + tempDir + ": " + e.getMessage()));
      return;
    }

    System.err.println("Wrote:\n" + workspaceFile + "\n" + buildFile);
  }

  private void cleanup() {
    resolver.cleanup();
    for (Event event : handler.getEvents()) {
      System.err.println(event);
    }
  }
}
