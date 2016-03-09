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

import com.google.common.io.Files;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixFileSystem;
import com.google.devtools.build.workspace.maven.Resolver;
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

  private final EventHandler handler;
  private final FileSystem fileSystem;
  private final Resolver resolver;
  private final Path outputDir;

  public static void main(String[] args) {
    OptionsParser parser = OptionsParser.newOptionsParser(GenerateWorkspaceOptions.class);
    parser.parseAndExitUponError(args);
    GenerateWorkspaceOptions options = parser.getOptions(GenerateWorkspaceOptions.class);
    if (options.mavenProjects.isEmpty()
        && options.bazelProjects.isEmpty()
        && options.artifacts.isEmpty()) {
      printUsage(parser);
      return;
    }

    GenerateWorkspace workspaceFileGenerator = new GenerateWorkspace(options.outputDir);
    workspaceFileGenerator.generateFromWorkspace(options.bazelProjects);
    workspaceFileGenerator.generateFromPom(options.mavenProjects);
    workspaceFileGenerator.generateFromArtifacts(options.artifacts);
    if (!workspaceFileGenerator.hasErrors()) {
      workspaceFileGenerator.writeResults();
    }
    workspaceFileGenerator.cleanup();
    if (workspaceFileGenerator.hasErrors()) {
      System.exit(1);
    }
  }

  private static void printUsage(OptionsParser parser) {
    System.out.println("Usage: generate_workspace (-b PATH|-m PATH|-a coord)+ [-o PATH]\n\n"
        + "Generates a WORKSPACE file from the given projects and a BUILD file with a rule that "
        + "contains all of the transitive dependencies. At least one bazel_project, "
        + "maven_project, or artifact coordinate must be specified. If output_dir is not "
        + "specified, the generated files will be written to a temporary directory.\n");
    System.out.println(parser.describeOptions(Collections.<String, String>emptyMap(),
        OptionsParser.HelpVerbosity.LONG));
  }

  private GenerateWorkspace(String outputDir) {
    this.handler = new EventHandler();
    this.fileSystem = getFileSystem();
    this.resolver = new Resolver(handler);
    if (outputDir.isEmpty()) {
      this.outputDir = fileSystem.getPath(Files.createTempDir().toString());
    } else {
      this.outputDir = fileSystem.getPath(outputDir);
    }
  }

  static FileSystem getFileSystem() {
    return OS.getCurrent() == OS.WINDOWS
        ? new JavaIoFileSystem() : new UnixFileSystem();
  }

  private void generateFromWorkspace(List<String> projects) {
    for (String project : projects) {
      WorkspaceResolver workspaceResolver = new WorkspaceResolver(resolver, handler);
      Path projectPath = fileSystem.getPath(getAbsolute(project));
      Package externalPackage = workspaceResolver.parse(projectPath.getRelative("WORKSPACE"));
      workspaceResolver.resolveTransitiveDependencies(externalPackage);
    }
  }

  private void generateFromPom(List<String> projects) {
    for (String project : projects) {
      resolver.resolvePomDependencies(getAbsolute(project));
    }
  }

  private void generateFromArtifacts(List<String> artifacts) {
    for (String artifactCoord : artifacts) {
      resolver.resolveArtifact(artifactCoord);
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
    File workspaceFile = outputDir.getRelative("WORKSPACE").getPathFile();
    File buildFile = outputDir.getRelative("BUILD").getPathFile();

    // Don't overwrite existing files with generated ones.
    if (workspaceFile.exists()) {
      workspaceFile = outputDir.getRelative(date + ".WORKSPACE").getPathFile();
    }
    if (buildFile.exists()) {
      buildFile = outputDir.getRelative(date + ".BUILD").getPathFile();
    }

    try (PrintStream workspaceStream = new PrintStream(workspaceFile);
         PrintStream buildStream = new PrintStream(buildFile)) {
      resolver.writeWorkspace(workspaceStream);
      resolver.writeBuild(buildStream);
    } catch (IOException e) {
      handler.handle(Event.error(
          "Could not write WORKSPACE and BUILD files to " + outputDir + ": " + e.getMessage()));
      return;
    }
    System.err.println("Wrote:\n" + workspaceFile + "\n" + buildFile);
  }

  private void cleanup() {
    for (Event event : handler.getEvents()) {
      System.err.println(event);
    }
  }

  private class EventHandler extends StoredEventHandler {
    @Override
    public void handle(Event event) {
      System.err.println(event.getKind() + ": " + event.getMessage());
    }
  }
}
