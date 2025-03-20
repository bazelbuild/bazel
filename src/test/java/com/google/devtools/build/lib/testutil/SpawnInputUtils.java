// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.base.MoreObjects.firstNonNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.ArtifactExpander.MissingExpansionException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import java.util.NoSuchElementException;

/** Utilities for finding {@link ActionInput} instances within a {@link Spawn}. */
public final class SpawnInputUtils {

  public static ActionInput getInputWithName(Spawn spawn, String name) {
    return spawn.getInputFiles().toList().stream()
        .filter(input -> input.getExecPathString().contains(name))
        .findFirst()
        .orElseThrow(() -> noSuchInput("spawn input", name, spawn));
  }

  public static ActionInput getFilesetInputWithName(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      String artifactName,
      String inputName) {
    for (ActionInput actionInput : spawn.getInputFiles().toList()) {
      if (!(actionInput instanceof Artifact fileset)
          || !fileset.isFileset()
          || !fileset.getExecPathString().contains(artifactName)) {
        continue;
      }
      for (FilesetOutputSymlink filesetOutputSymlink :
          inputMetadataProvider.getFileset(fileset).symlinks()) {
        if (filesetOutputSymlink.targetPath().getPathString().contains(inputName)) {
          return ActionInputHelper.fromPath(filesetOutputSymlink.targetPath());
        }
      }
    }
    throw noSuchInput("fileset input in " + artifactName, inputName, spawn);
  }

  public static ActionInput getRunfilesFilesetInputWithName(
      Spawn spawn, ActionExecutionContext context, String artifactName, String inputName) {
    Artifact filesetArtifact = getRunfilesArtifactWithName(spawn, context, artifactName);
    checkState(filesetArtifact.isFileset(), filesetArtifact);

    ImmutableList<FilesetOutputSymlink> filesetLinks;
    try {
      filesetLinks = context.getArtifactExpander().expandFileset(filesetArtifact).symlinks();
    } catch (MissingExpansionException e) {
      throw new IllegalStateException(e);
    }
    for (FilesetOutputSymlink filesetOutputSymlink : filesetLinks) {
      if (filesetOutputSymlink.targetPath().toString().contains(inputName)) {
        return ActionInputHelper.fromPath(filesetOutputSymlink.targetPath());
      }
    }
    throw noSuchInput("runfiles fileset in " + filesetArtifact, inputName, spawn);
  }

  public static SpecialArtifact getTreeArtifactWithName(Spawn spawn, String name) {
    ActionInput input = getInputWithName(spawn, name);
    checkState(
        input instanceof SpecialArtifact && ((SpecialArtifact) input).isTreeArtifact(),
        "Expected spawn %s to have tree artifact input with name %s, but it is: %s",
        spawn.getResourceOwner().describe(),
        name,
        input);
    return (SpecialArtifact) input;
  }

  public static Artifact getExpandedToArtifact(
      String name, Artifact expandableArtifact, Spawn spawn, ActionExecutionContext context) {
    return context.getArtifactExpander().tryExpandTreeArtifact(expandableArtifact).stream()
        .filter(artifact -> artifact.getExecPathString().contains(name))
        .findFirst()
        .orElseThrow(
            () -> noSuchInput("artifact expanded from " + expandableArtifact, name, spawn));
  }

  public static Artifact getRunfilesArtifactWithName(
      Spawn spawn, ActionExecutionContext context, String name) {
    return spawn.getInputFiles().toList().stream()
        .filter(i -> i instanceof Artifact && ((Artifact) i).isRunfilesTree())
        .map(i -> context.getInputMetadataProvider().getRunfilesMetadata(i).getRunfilesTree())
        .flatMap(t -> t.getArtifacts().toList().stream())
        .filter(artifact -> artifact.getExecPathString().contains(name))
        .findFirst()
        .orElseThrow(() -> noSuchInput("runfiles artifact", name, spawn));
  }

  private static NoSuchElementException noSuchInput(String inputType, String name, Spawn spawn) {
    ActionExecutionMetadata action = spawn.getResourceOwner();
    return new NoSuchElementException(
        String.format(
            "No %s named %s in %s",
            inputType, name, firstNonNull(action.getProgressMessage(), action.prettyPrint())));
  }

  private SpawnInputUtils() {}
}
