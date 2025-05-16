// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.MissingDepExecException;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * An {@link InputMetadataProvider} implementation that requests the metadata of derived artifacts
 * from Skyframe and that of source artifacts from the per-build metadata cache.
 *
 * <p>During input discovery, the action may well legally read scheduling dependencies that are not
 * also inputs. Those are not in the regular input metadata provider (doing so would be a
 * performance issue), so we need to ask those from Skyframe instead. It's not that problematic
 * because they are known to be transitive Skyframe deps, so we can rely on them being present, with
 * one exception (see below) that can be handled without much ceremony.
 *
 * <p>In theory, this would also work for source artifacts. However, the performance ramifications
 * of doing that are unknown.
 */
final class SkyframeInputMetadataProvider implements InputMetadataProvider {
  // Not static since it uses "env" and "envMonitor". This works out because "env" is always
  // updated to the Environment instance from the last restart and SkyframeLookup closes over
  // SkyframeInputMetadataProvider and not "env".
  private class SkyframeLookup {
    private final SkyKey key;
    private volatile SkyValue value;

    public SkyframeLookup(SkyKey key) {
      this.key = key;
      this.value = null;
    }

    private SkyValue tryLookup() throws InterruptedException, MissingDepExecException {
      if (value != null) {
        return value;
      }

      // We reuse envMonitor to guard SkyframeLookup.value. It's the simplest: we need a lock for
      // "env" that ensures that only one thread calls methods on it and thus a simple
      // synchronized" block won't work so if we wanted to guard "value" separately, we'd have to
      // have two nested "synchronized" blocks which sounds like more trouble than it's worth.
      synchronized (envMonitor) {
        // We use .getExistingValue() to spare a Skyframe edge. This is correct because these are
        // always transitive dependencies (it's not a property that's inherently true, though, it's
        // just how actions that discover inputs happen to be implemented)
        SkyValue localValue = evaluator.getExistingValue(key);
        if (localValue == null) {
          // This can only happen if a transitive dependency was rewound but the re-evaluation
          // resulted in an error or the rewinding is in progress.
          //
          // env is set to null once any missing values are detected. This is a work-around for a
          // semi-bug in include scanning where the include scanner might continue processing after
          // the action has already ended. This is problematic because lookups against an
          // environment crash once the associated action is done. Instead, any subsequent lookups
          // throw MissingDepExecException without adding a dependency edge. At worst, this can only
          // result in a superfluous restart.
          if (env == null) {
            throw new MissingDepExecException();
          }
          localValue = env.getValue(key);
          if (localValue == null) {
            env = null;
            throw new MissingDepExecException();
          }
          // This can happen if the evaluation of "value" finished between the getExistingValue()
          // call and the getValue() one. In this case, "value" is good. We move on.
        }

        value = localValue;
        return localValue;
      }
    }
  }

  private final MemoizingEvaluator evaluator;

  // Non-null while skyframe lookups are being allowed during input discovery.
  @Nullable private SkyFunction.Environment env;

  private final Object envMonitor;
  private final InputMetadataProvider perBuild;
  private final PathFragment relativeOutputPath;

  private final ConcurrentHashMap<String, ActionInput> seen;

  /**
   * A cache so that we don't need to look up any SkyValue twice.
   *
   * <p>This is necessary because action rewinding means that even though a {@code getValue()} call
   * returned the appropriate value alright, subsequent calls with the same {@code SkyKey} may not
   * do so. So theoretically, every call to {@link #getInputMetadata(ActionInput)} should be
   * prepared to handle a {@code MissingDepExecException}.
   *
   * <p>Sadly, that's not the case and the invariant we have is that the <b>first</b> call over the
   * course of the evaluation of an action with any given {@code ActionInput} handles that case, the
   * subsequent ones not necessarily. This cache is there to make sure that that's alright.
   */
  private final ConcurrentHashMap<SkyKey, SkyframeLookup> skyframeLookups;

  private boolean allowSkyframe;

  SkyframeInputMetadataProvider(
      MemoizingEvaluator evaluator, InputMetadataProvider perBuild, String relativeOutputPath) {
    this.evaluator = evaluator;
    this.envMonitor = new Object();
    this.perBuild = perBuild;
    this.relativeOutputPath = PathFragment.create(relativeOutputPath);
    this.seen = new ConcurrentHashMap<>();
    this.skyframeLookups = new ConcurrentHashMap<>();
    this.allowSkyframe = false;
  }

  /**
   * Allow Skyframe access while the returned closeable is open.
   *
   * <p>This should only happen during input discovery, so we disallow it everywhere else.
   */
  // TODO: b/416449869 - Add test coverage for a new env being set after a skyframe restart.
  SilentCloseable withSkyframeAllowed(SkyFunction.Environment env) {
    // No need to synchronize with envMonitor here. This is called before input discovery begins,
    // and the closeable is called after input discovery ends, so there are no concurrent calls to
    // getInputMetadataChecked.
    allowSkyframe = true;
    this.env = env;
    return () -> {
      allowSkyframe = false;
      this.env = null;
    };
  }

  @Nullable
  @Override
  public FileArtifactValue getInputMetadataChecked(ActionInput input)
      throws InterruptedException, IOException, MissingDepExecException {
    if (!(input instanceof Artifact artifact)) {
      if (!input.getExecPath().startsWith(relativeOutputPath)) {
        return perBuild.getInputMetadataChecked(input);
      } else {
        return null;
      }
    }

    if (artifact.isSourceArtifact()) {
      return perBuild.getInputMetadataChecked(input);
    }

    if (!allowSkyframe) {
      return null;
    }

    if (artifact instanceof SpecialArtifact) {
      return null;
    }

    SkyKey key = Artifact.key(artifact);
    SkyframeLookup lookup = skyframeLookups.computeIfAbsent(key, SkyframeLookup::new);
    SkyValue value = lookup.tryLookup();
    seen.put(artifact.getExecPathString(), artifact);
    ActionExecutionValue actionExecutionValue = (ActionExecutionValue) value;
    return actionExecutionValue.getExistingFileArtifactValue(artifact);
  }

  @Nullable
  @Override
  public TreeArtifactValue getTreeMetadata(ActionInput input) {
    return null;
  }

  @Nullable
  @Override
  public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
    return null;
  }

  @Nullable
  @Override
  public FilesetOutputTree getFileset(ActionInput input) {
    return null;
  }

  @Override
  public Map<Artifact, FilesetOutputTree> getFilesets() {
    return ImmutableMap.of();
  }

  @Nullable
  @Override
  public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
    return null;
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public ActionInput getInput(String execPath) {
    ActionInput result = seen.get(execPath);
    if (result == null) {
      result = perBuild.getInput(execPath);
    }

    return result;
  }
}
