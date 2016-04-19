
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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Map;
import java.util.SortedMap;

/**
 * This action creates a set of symbolic links.
 */
@Immutable
public final class CreateIncSymlinkAction extends AbstractAction {
  private final ImmutableSortedMap<Artifact, Artifact> symlinks;

  /**
   * Creates a new instance. The symlinks map maps symlinks to their targets, i.e. the symlink paths
   * must be unique, but several of them can point to the same target.
   */
  public CreateIncSymlinkAction(ActionOwner owner, Map<Artifact, Artifact> symlinks) {
    super(owner, ImmutableList.copyOf(symlinks.values()), ImmutableList.copyOf(symlinks.keySet()));
    this.symlinks = ImmutableSortedMap.copyOf(symlinks, Artifact.EXEC_PATH_COMPARATOR);
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
  throws ActionExecutionException {
    try {
      for (Map.Entry<Artifact, Artifact> entry : symlinks.entrySet()) {
        Path symlink = entry.getKey().getPath();
        symlink.createSymbolicLink(entry.getValue().getPath());
      }
    } catch (IOException e) {
      String message = "IO Error while creating symlink";
      throw new ActionExecutionException(message, e, this, false);
    }
  }

  @VisibleForTesting
  public SortedMap<Artifact, Artifact> getSymlinks() {
    return symlinks;
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    // We're mainly doing I/O, so CPU usage should be very low; most of the
    // time we'll be blocked waiting for the OS.
    // The only exception is the fingerprint digest calculation for the stamp
    // file contents.
    return ResourceSet.createWithRamCpuIo(/*memoryMb=*/0, /*cpuUsage=*/0.005, /*ioUsage=*/0.0);
  }

  @Override
  public String computeKey() {
    Fingerprint key = new Fingerprint();
    for (Map.Entry<Artifact, Artifact> entry : symlinks.entrySet()) {
      key.addPath(entry.getKey().getPath());
      key.addPath(entry.getValue().getPath());
    }
    return key.hexDigestAndReset();
  }

  @Override
  protected String getRawProgressMessage() {
    return null; // users don't really want to know about inc symlinks.
  }

  @Override
  public String getMnemonic() {
    return "Symlink";
  }
}

