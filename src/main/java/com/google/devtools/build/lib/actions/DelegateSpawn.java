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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.extra.SpawnInfo;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;

/**
 * A delegating spawn that allow us to overwrite certain methods while maintaining the original
 * behavior for non-overwritten methods.
 */
public class DelegateSpawn implements Spawn {

  private final Spawn spawn;

  public DelegateSpawn(Spawn spawn){
    this.spawn = spawn;
  }

  @Override
  public final ImmutableMap<String, String> getExecutionInfo() {
    return spawn.getExecutionInfo();
  }

  @Override
  public boolean isRemotable() {
    return spawn.isRemotable();
  }

  @Override
  public ImmutableList<Artifact> getFilesetManifests() {
    return spawn.getFilesetManifests();
  }

  @Override
  public String asShellCommand(Path workingDir) {
    return spawn.asShellCommand(workingDir);
  }

  @Override
  public ImmutableMap<PathFragment, Artifact> getRunfilesManifests() {
    return spawn.getRunfilesManifests();
  }

  @Override
  public RunfilesSupplier getRunfilesSupplier() {
    return spawn.getRunfilesSupplier();
  }

  @Override
  public SpawnInfo getExtraActionInfo() {
    return spawn.getExtraActionInfo();
  }

  @Override
  public ImmutableList<String> getArguments() {
    return spawn.getArguments();
  }

  @Override
  public ImmutableMap<String, String> getEnvironment() {
    return spawn.getEnvironment();
  }

  @Override
  public Iterable<? extends ActionInput> getToolFiles() {
    return spawn.getToolFiles();
  }

  @Override
  public Iterable<? extends ActionInput> getInputFiles() {
    return spawn.getInputFiles();
  }

  @Override
  public Collection<? extends ActionInput> getOutputFiles() {
    return spawn.getOutputFiles();
  }

  @Override
  public Collection<PathFragment> getOptionalOutputFiles() {
    return spawn.getOptionalOutputFiles();
  }

  @Override
  public ActionMetadata getResourceOwner() {
    return spawn.getResourceOwner();
  }

  @Override
  public ResourceSet getLocalResources() {
    return spawn.getLocalResources();
  }

  @Override
  public ActionOwner getOwner() {
    return spawn.getOwner();
  }

  @Override
  public String getMnemonic() {
    return spawn.getMnemonic();
  }
}
