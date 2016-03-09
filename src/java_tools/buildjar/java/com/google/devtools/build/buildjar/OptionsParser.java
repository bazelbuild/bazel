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

package com.google.devtools.build.buildjar;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * Parses options that the {@link JavaLibraryBuildRequest} needs to construct a build request from
 * command-line flags and options files and provides them via getters.
 */
public final class OptionsParser {
  private final List<String> javacOpts = new ArrayList<>();

  private final Map<String, String> directJarsToTargets = new HashMap<>();
  private final Map<String, String> indirectJarsToTargets = new HashMap<>();

  private String strictJavaDeps;

  private String outputDepsFile;
  private String outputDepsProtoFile;
  private final Set<String> depsArtifacts = new HashSet<>();

  private boolean strictClasspathMode;

  private String sourceGenDir;
  private String generatedSourcesOutputJar;
  private String manifestProtoPath;
  private final Set<String> sourceRoots = new HashSet<>();

  private final List<String> sourceFiles = new ArrayList<>();
  private final List<String> sourceJars = new ArrayList<>();
  private final List<String> messageFiles = new ArrayList<>();
  private final List<String> resourceFiles = new ArrayList<>();
  private final List<String> resourceJars = new ArrayList<>();
  private final List<String> rootResourceFiles = new ArrayList<>();

  private String classPath = "";

  private String extdir;

  private String processorPath = "";
  private final List<String> processorNames = new ArrayList<>();

  private String outputJar;

  private String classDir;
  private String tempDir;

  private final Map<String, List<String>> postProcessors = new LinkedHashMap<>();

  private boolean compressJar;

  private String ruleKind;
  private String targetLabel;

  /**
   * Constructs an {@code OptionsParser} from a list of command args. Sets the same
   * JavacRunner for both compilation and annotation processing.
   *
   * @param args the list of command line args.
   * @throws InvalidCommandLineException on any command line error.
   */
  public OptionsParser(List<String> args) throws InvalidCommandLineException, IOException {
    processCommandlineArgs(expandArguments(args));
  }

  /**
   * Processes the command line arguments.
   *
   * @throws InvalidCommandLineException on an invalid option being passed.
   */
  private void processCommandlineArgs(Deque<String> argQueue) throws InvalidCommandLineException {
    for (String arg = argQueue.pollFirst(); arg != null; arg = argQueue.pollFirst()) {
      switch (arg) {
        case "--javacopts":
          // Collect additional arguments to javac.
          // Assumes that javac options do not start with "--".
          // otherwise we have to do something like adding a "--"
          // terminator to the passed arguments.
          collectFlagArguments(javacOpts, argQueue, "--");
          break;
        case "--direct_dependency":
          {
            String jar = getArgument(argQueue, arg);
            String target = getArgument(argQueue, arg);
            directJarsToTargets.put(jar, target);
            break;
          }
        case "--indirect_dependency":
          {
            String jar = getArgument(argQueue, arg);
            String target = getArgument(argQueue, arg);
            indirectJarsToTargets.put(jar, target);
            break;
          }
        case "--strict_java_deps":
          strictJavaDeps = getArgument(argQueue, arg);
          break;
        case "--output_deps":
          outputDepsFile = getArgument(argQueue, arg);
          break;
        case "--output_deps_proto":
          outputDepsProtoFile = getArgument(argQueue, arg);
          break;
        case "--deps_artifacts":
          collectFlagArguments(depsArtifacts, argQueue, "--");
          break;
        case "--reduce_classpath":
          strictClasspathMode = true;
          break;
        case "--sourcegendir":
          sourceGenDir = getArgument(argQueue, arg);
          break;
        case "--generated_sources_output":
          generatedSourcesOutputJar = getArgument(argQueue, arg);
          break;
        case "--output_manifest_proto":
          manifestProtoPath = getArgument(argQueue, arg);
          break;
        case "--source_roots":
          collectFlagArguments(sourceRoots, argQueue, "-");
          break;
        case "--sources":
          collectFlagArguments(sourceFiles, argQueue, "-");
          break;
        case "--source_jars":
          collectFlagArguments(sourceJars, argQueue, "-");
          break;
        case "--messages":
          collectFlagArguments(messageFiles, argQueue, "-");
          break;
        case "--resources":
          collectFlagArguments(resourceFiles, argQueue, "-");
          break;
        case "--resource_jars":
          collectFlagArguments(resourceJars, argQueue, "-");
          break;
        case "--classpath_resources":
          collectFlagArguments(rootResourceFiles, argQueue, "-");
          break;
        case "--classpath":
          classPath = getArgument(argQueue, arg);
          break;
        case "--processorpath":
          processorPath = getArgument(argQueue, arg);
          break;
        case "--processors":
          collectProcessorArguments(processorNames, argQueue, "-");
          break;
        case "--extdir":
          extdir = getArgument(argQueue, arg);
          break;
        case "--output":
          outputJar = getArgument(argQueue, arg);
          break;
        case "--classdir":
          classDir = getArgument(argQueue, arg);
          break;
        case "--tempdir":
          tempDir = getArgument(argQueue, arg);
          break;
        case "--gendir":
          // TODO(bazel-team) - remove when Bazel no longer passes this flag to buildjar.
          getArgument(argQueue, arg);
          break;
        case "--post_processor":
          addExternalPostProcessor(argQueue, arg);
          break;
        case "--compress_jar":
          compressJar = true;
          break;
        case "--rule_kind":
          ruleKind = getArgument(argQueue, arg);
          break;
        case "--target_label":
          targetLabel = getArgument(argQueue, arg);
          break;
        default:
          throw new InvalidCommandLineException("unknown option : '" + arg + "'");
      }
    }
  }

  /**
   * Pre-processes an argument list, expanding options @filename to read in
   * the content of the file and add it to the list of arguments.
   *
   * @param args the List of arguments to pre-process.
   * @return the List of pre-processed arguments.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  private static Deque<String> expandArguments(List<String> args) throws IOException {
    Deque<String> expanded = new ArrayDeque<>(args.size());
    for (String arg : args) {
      expandArgument(expanded, arg);
    }
    return expanded;
  }

  /**
   * Expands a single argument, expanding options @filename to read in the content of the file
   * and add it to the list of processed arguments. The @ itself can be escaped with @@.
   *
   * @param expanded the list of processed arguments.
   * @param arg the argument to pre-process.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  private static void expandArgument(Deque<String> expanded, String arg) throws IOException {
    if (arg.startsWith("@") && !arg.startsWith("@@")) {
      for (String line : Files.readAllLines(Paths.get(arg.substring(1)), UTF_8)) {
        if (line.length() > 0) {
          expandArgument(expanded, line);
        }
      }
    } else {
      expanded.add(arg);
    }
  }

  /**
   * Collects the arguments for a command line flag until it finds a flag that starts with the
   * terminatorPrefix.
   *
   * @param output where to put the collected flag arguments.
   * @param args
   * @param terminatorPrefix the terminator prefix to stop collecting of argument flags.
   */
  private static void collectFlagArguments(
      Collection<String> output, Deque<String> args, String terminatorPrefix) {
    for (String arg = args.pollFirst(); arg != null; arg = args.pollFirst()) {
      if (arg.startsWith(terminatorPrefix)) {
        args.addFirst(arg);
        break;
      }
      output.add(arg);
    }
  }

  /**
   * Collects the arguments for the --processors command line flag until it finds a flag that
   * starts with the terminatorPrefix.
   *
   * @param output where to put the collected flag arguments.
   * @param args
   * @param terminatorPrefix the terminator prefix to stop collecting of argument flags.
   */
  private static void collectProcessorArguments(List<String> output, Deque<String> args,
      String terminatorPrefix) throws InvalidCommandLineException {
    for (String arg = args.pollFirst(); arg != null; arg = args.pollFirst()) {
      if (arg.startsWith(terminatorPrefix)) {
        args.addFirst(arg);
        break;
      }
      if (arg.contains(",")) {
        throw new InvalidCommandLineException("processor argument may not contain commas: " + arg);
      }
      output.add(arg);
    }
  }

  private static String getArgument(Deque<String> args, String arg)
      throws InvalidCommandLineException {
    try {
      return args.remove();
    } catch (NoSuchElementException e) {
      throw new InvalidCommandLineException(arg + ": missing argument");
    }
  }

  private void addExternalPostProcessor(Deque<String> args, String arg)
      throws InvalidCommandLineException {
    String processorName = getArgument(args, arg);
    List<String> arguments = new ArrayList<>();
    collectFlagArguments(arguments, args, "--");
    postProcessors.put(processorName, arguments);
  }

  public List<String> getJavacOpts() {
    return javacOpts;
  }

  public Map<String, String> getDirectMappings() {
    return directJarsToTargets;
  }

  public Map<String, String> getIndirectMappings() {
    return indirectJarsToTargets;
  }

  public String getStrictJavaDeps() {
    return strictJavaDeps;
  }

  public String getOutputDepsFile() {
    return outputDepsFile;
  }

  public String getOutputDepsProtoFile() {
    return outputDepsProtoFile;
  }

  public Set<String> getDepsArtifacts() {
    return depsArtifacts;
  }

  public boolean reduceClasspath() {
    return strictClasspathMode;
  }

  public String getSourceGenDir() {
    return sourceGenDir;
  }

  public String getGeneratedSourcesOutputJar() {
    return generatedSourcesOutputJar;
  }

  public String getManifestProtoPath() {
    return manifestProtoPath;
  }

  public Set<String> getSourceRoots() {
    return sourceRoots;
  }

  public List<String> getSourceFiles() {
    return sourceFiles;
  }

  public List<String> getSourceJars() {
    return sourceJars;
  }

  public List<String> getMessageFiles() {
    return messageFiles;
  }

  public List<String> getResourceFiles() {
    return resourceFiles;
  }

  public List<String> getResourceJars() {
    return resourceJars;
  }

  public List<String> getRootResourceFiles() {
    return rootResourceFiles;
  }

  public String getClassPath() {
    return classPath;
  }

  public String getExtdir() {
    return extdir;
  }

  public String getProcessorPath() {
    return processorPath;
  }

  public List<String> getProcessorNames() {
    return processorNames;
  }

  public String getOutputJar() {
    return outputJar;
  }

  public String getClassDir() {
    return classDir;
  }

  public String getTempDir() {
    return tempDir;
  }

  public Map<String, List<String>> getPostProcessors() {
    return postProcessors;
  }

  public boolean compressJar() {
    return compressJar;
  }

  public String getRuleKind() {
    return ruleKind;
  }

  public String getTargetLabel() {
    return targetLabel;
  }
}
