// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.IncludeProcessing;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScannerSupplier;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Returns all inclusions that were discovered by the header scanner tool to implement the header
 * thinning feature.
 *
 * <p>Reads the .headers_list output file if one was generated for the actions source file and
 * returns the Artifact objects associated with the headers that were found.
 */
@AutoCodec
public class HeaderThinning implements IncludeProcessing {
  private final Iterable<Artifact> potentialInputs;

  public HeaderThinning(Iterable<Artifact> potentialInputs) {
    // Just store this, don't create map of potential inputs at construction so that it is done
    // later at execution time rather than analysis time when this is instantiated.
    this.potentialInputs = potentialInputs;
  }

  private Map<PathFragment, Artifact> getAllowedInputsMap() {
    Map<PathFragment, Artifact> allowedInputsMap = new HashMap<>();
    for (Artifact input : potentialInputs) {
      allowedInputsMap.put(input.getExecPath(), input);
    }
    return allowedInputsMap;
  }

  @Nullable
  private static Artifact findHeadersListFile(NestedSet<Artifact> artifacts) {
    for (Artifact artifact : artifacts) {
      if (artifact.getExtension().equals("headers_list")) {
        return artifact;
      }
    }
    return null;
  }

  @Override
  public Iterable<Artifact> determineAdditionalInputs(
      @Nullable IncludeScannerSupplier includeScannerSupplier,
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeScanningHeaderData includeScanningHeaderData)
      throws ExecException {
    Artifact headersListFile = findHeadersListFile(action.getMandatoryInputs());
    if (headersListFile == null) {
      return null;
    }
    return findRequiredHeaderInputs(action.getSourceFile(), headersListFile, getAllowedInputsMap(),
        actionExecutionContext == null
            ? ArtifactPathResolver.IDENTITY
            : actionExecutionContext.getPathResolver());
  }

  /**
   * Reads the header scanning output file and discovers all of those headers as input artifacts.
   *
   * @param sourceFile the source that requires these headers
   * @param headersListFile .headers_list file output from header_scanner tool to be read
   * @param inputArtifactsMap map of PathFragment to Artifact of possible headers
   * @param pathResolver used to read the headersListFile
   * @return collection of header artifacts that are required for {@code action} to compile
   * @throws ExecException on environmental (IO) or user errors
   */
  @VisibleForTesting
  static Iterable<Artifact> findRequiredHeaderInputs(
      Artifact sourceFile, Artifact headersListFile, Map<PathFragment, Artifact> inputArtifactsMap,
      ArtifactPathResolver pathResolver)
      throws ExecException {
    try {
      ImmutableList.Builder<Artifact> includeBuilder = ImmutableList.builder();
      List<PathFragment> missing = new ArrayList<>();
      for (String line :
          FileSystemUtils.readLines(pathResolver.toPath(headersListFile), StandardCharsets.UTF_8)) {
        if (line.isEmpty()) {
          continue;
        }

        PathFragment headerPath = PathFragment.create(line);
        Artifact header = inputArtifactsMap.get(headerPath);
        if (header == null) {
          missing.add(headerPath);
        } else {
          includeBuilder.add(header);
        }
      }

      if (!missing.isEmpty()) {
        includeBuilder.addAll(
            findRequiredHeaderInputsInTreeArtifacts(sourceFile, inputArtifactsMap, missing));
      }
      return includeBuilder.build();
    } catch (IOException ex) {
      throw new EnvironmentalExecException(
          String.format("Error reading headers file %s", headersListFile.getExecPathString()), ex);
    }
  }

  /**
   * Headers inside a TreeArtifact will not have their ExecPath as a key in the map as they do not
   * have their own Artifact object. These headers must be mapped to their containing TreeArtifact.
   * We are unable to select individual files from within a TreeArtifact so must discover the entire
   * TreeArtifact as an input.
   */
  private static Iterable<Artifact> findRequiredHeaderInputsInTreeArtifacts(
      Artifact sourceFile,
      Map<PathFragment, Artifact> inputArtifactsMap,
      List<PathFragment> missing)
      throws ExecException {
    ImmutableList.Builder<Artifact> includeBuilder = ImmutableList.builder();
    ImmutableList.Builder<PathFragment> treeArtifactPathsBuilder = ImmutableList.builder();
    for (Map.Entry<PathFragment, Artifact> inputEntry : inputArtifactsMap.entrySet()) {
      if (inputEntry.getValue().isTreeArtifact()) {
        treeArtifactPathsBuilder.add(inputEntry.getKey());
      }
    }

    ImmutableList<PathFragment> treeArtifactPaths = treeArtifactPathsBuilder.build();
    for (PathFragment missingPath : missing) {
      includeBuilder.add(
          findRequiredHeaderInputInTreeArtifacts(
              sourceFile, treeArtifactPaths, inputArtifactsMap, missingPath));
    }
    return includeBuilder.build();
  }

  private static Artifact findRequiredHeaderInputInTreeArtifacts(
      Artifact sourceFile,
      List<PathFragment> treeArtifactPaths,
      Map<PathFragment, Artifact> inputArtifactsMap,
      PathFragment missingPath)
      throws ExecException {
    for (PathFragment treeArtifactPath : treeArtifactPaths) {
      if (missingPath.startsWith(treeArtifactPath)) {
        return inputArtifactsMap.get(treeArtifactPath);
      }
    }

    throw new UserExecException(
        String.format(
            "Unable to map header file (%s) found during header scanning of %s."
                + " This is usually the result of a case mismatch",
            missingPath, sourceFile.getExecPathString()));
  }
}
