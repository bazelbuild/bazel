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
package com.google.devtools.build.xcode.plmerge;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.xcode.plmerge.PlMerge.PlMergeOptions;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos.Control;

import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Container for data consumed by plmerge
 */
class MergingArguments {

  private final FileSystem fileSystem = FileSystems.getDefault();
  private final List<Path> sourceFilePaths;
  private final List<Path> immutableSourceFilePaths;
  private final String outFile;
  private final Map<String, String> variableSubstitutions;
  private final String primaryBundleId;
  private final String fallbackBundleId;

  /**
   * Build MergingArguments from a plmerge protobuf.
   */
  public MergingArguments(Control control) {
    ImmutableList.Builder<Path> sourceFilePathsBuilder = new Builder<>();
    for (String pathString : control.getSourceFileList()) {
      sourceFilePathsBuilder.add(fileSystem.getPath(pathString));
    }
    sourceFilePaths = sourceFilePathsBuilder.build();
    
    ImmutableList.Builder<Path> immutableSourceFilePathsBuilder = new Builder<>();
    for (String pathString : control.getImmutableSourceFileList()) {
      immutableSourceFilePathsBuilder.add(fileSystem.getPath(pathString));
    }
    immutableSourceFilePaths = immutableSourceFilePathsBuilder.build();
    
    outFile = control.getOutFile();
    variableSubstitutions = control.getVariableSubstitutionMap();
    primaryBundleId = control.getPrimaryBundleId();
    fallbackBundleId = control.getFallbackBundleId();
  }

  /**
   * Build MergingArguments from command line arguments passed to the plmerge executable.
   */
  public MergingArguments(PlMergeOptions options) {
    ImmutableList.Builder<Path> sourceFilePathsBuilder = new Builder<Path>();
    for (String sourceFile : options.sourceFiles) {
      sourceFilePathsBuilder.add(fileSystem.getPath(sourceFile));
    }

    sourceFilePaths = sourceFilePathsBuilder.build();
    immutableSourceFilePaths = ImmutableList.<Path>of();
    outFile = options.outFile;
    variableSubstitutions = ImmutableMap.<String, String>of();
    primaryBundleId = options.primaryBundleId;
    fallbackBundleId = options.fallbackBundleId;
  }

  /**
   * Returns paths to the plist files to merge relative to plmerge. These can be
   * binary, XML, or ASCII format.
   */
  public List<Path> getSourceFilePaths() {
    return sourceFilePaths;
  }

  /*
   * Returns paths to plist files with keys which may not be overwritten.
   */
  public List<Path> getImmutableSourceFilePaths() {
    return immutableSourceFilePaths;
  }
  
  /**
   * Returns path to the output file to merge relative to plmerge.
   */
  public String getOutFile() {
    return outFile;
  }

  /**
   * Returns a reverse-DNS string identifier for this bundle associated with output
   * binary plist.  Overrides the bundle id specified in the CFBundleIdentifier
   * plist field.
   */
  public String getPrimaryBundleId() {
    return primaryBundleId;
  }

  /**
   * Returns a fallback reverse-DNS string identifier for this bundle when bundle
   * identifier is not specified in primary_bundle_id or an associated plist
   * file.
   */
  public String getFallbackBundleId() {
    return fallbackBundleId;
  }

  /**
   * Returns key-value substitutions to support templating for plists.  A substitution
   * is made if the substitution key appears as a value for any key-value pair
   * in any source_file.
   * For example, a plist with the entry:
   *    <pre><key>CFBundleExectuable</key>
   *    <string>EXECUTABLE_NAME</string></pre>
   * could be templated by passing a variable substitution like
   *    {"EXECUTABLE_NAME", "PrenotCalculator"}
   */
  public Map<String, String> getVariableSubstitutions() {
    return variableSubstitutions;
  }
}
