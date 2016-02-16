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
package com.google.devtools.build.lib.actions.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.AbstractActionOwner;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.AbstractSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrExceptionUtils;
import com.google.devtools.build.skyframe.ValueOrUntypedException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A bunch of utilities that are useful for test concerning actions, artifacts,
 * etc.
 */
public final class ActionsTestUtil {

  private final ActionGraph actionGraph;

  public ActionsTestUtil(ActionGraph actionGraph) {
    this.actionGraph = actionGraph;
  }

  private static final Label NULL_LABEL = Label.parseAbsoluteUnchecked("//null/action:owner");

  public static ActionExecutionContext createContext(Executor executor, FileOutErr fileOutErr,
      Path execRoot, MetadataHandler metadataHandler, @Nullable ActionGraph actionGraph) {
    return new ActionExecutionContext(
        executor,
        new SingleBuildFileCache(execRoot.getPathString(), execRoot.getFileSystem()),
        metadataHandler, fileOutErr,
        actionGraph == null
            ? null
            : ActionInputHelper.actionGraphArtifactExpander(actionGraph));
  }

  public static ActionExecutionContext createContextForInputDiscovery(Executor executor,
      FileOutErr fileOutErr, Path execRoot, MetadataHandler metadataHandler,
      BuildDriver buildDriver) {
    return ActionExecutionContext.forInputDiscovery(
        executor,
        new SingleBuildFileCache(execRoot.getPathString(), execRoot.getFileSystem()),
        metadataHandler, fileOutErr,
        new BlockingSkyFunctionEnvironment(buildDriver,
            executor == null ? null : executor.getEventHandler()));

  }


  /**
   * {@link SkyFunction.Environment} that internally makes a full Skyframe evaluate call for the
   * requested keys, blocking until the values are ready.
   */
  private static class BlockingSkyFunctionEnvironment extends AbstractSkyFunctionEnvironment {
    private final BuildDriver driver;
    private final EventHandler eventHandler;

    private BlockingSkyFunctionEnvironment(BuildDriver driver, EventHandler eventHandler) {
      this.driver = driver;
      this.eventHandler = eventHandler;
    }

    @Override
    protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
        Set<SkyKey> depKeys) {
      EvaluationResult<SkyValue> evaluationResult;
      Map<SkyKey, ValueOrUntypedException> result = Maps.newHashMapWithExpectedSize(depKeys.size());
      try {
        evaluationResult = driver.evaluate(depKeys, /*keepGoing=*/false,
            ResourceUsage.getAvailableProcessors(), eventHandler);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        for (SkyKey key : depKeys) {
          result.put(key, ValueOrExceptionUtils.ofNull());
        }
        return result;
      }
      for (SkyKey key : depKeys) {
        SkyValue value = evaluationResult.get(key);
        if (value != null) {
          result.put(key, ValueOrExceptionUtils.ofValue(value));
          continue;
        }
        ErrorInfo errorInfo = evaluationResult.getError(key);
        if (errorInfo == null || errorInfo.getException() == null) {
          result.put(key, ValueOrExceptionUtils.ofNull());
          continue;
        }
        result.put(key, ValueOrExceptionUtils.ofExn(errorInfo.getException()));
      }
      return result;
    }

    @Override
    public EventHandler getListener() {
      return null;
    }

    @Override
    public boolean inErrorBubblingForTesting() {
      return false;
    }
  }

  /**
   * A dummy ActionOwner implementation for use in tests.
   */
  public static class NullActionOwner extends AbstractActionOwner {
    @Override
    public Label getLabel() {
      return NULL_LABEL;
    }

    @Override
    public String getConfigurationMnemonic() {
      return "dummy-configuration-mnemonic";
    }

    @Override
    public final String getConfigurationChecksum() {
      return "dummy-configuration";
    }
  }

  public static final Artifact DUMMY_ARTIFACT = new Artifact(
      new PathFragment("dummy"),
      Root.asSourceRoot(new InMemoryFileSystem().getRootDirectory()));

  public static final ActionOwner NULL_ACTION_OWNER = new NullActionOwner();

  public static final ArtifactOwner NULL_ARTIFACT_OWNER =
      new ArtifactOwner() {
        @Override
        public Label getLabel() {
          return NULL_LABEL;
        }
  };

  public static class UncheckedActionConflictException extends RuntimeException {
    public UncheckedActionConflictException(ActionConflictException e) {
      super(e);
    }
  }

  /**
   * A dummy Action class for use in tests.
   */
  public static class NullAction extends AbstractAction {

    public NullAction() {
      super(NULL_ACTION_OWNER, Artifact.NO_ARTIFACTS, ImmutableList.of(DUMMY_ARTIFACT));
    }

    public NullAction(ActionOwner owner, Artifact... outputs) {
      super(owner, Artifact.NO_ARTIFACTS, ImmutableList.copyOf(outputs));
    }

    public NullAction(Artifact... outputs) {
      super(NULL_ACTION_OWNER, Artifact.NO_ARTIFACTS, ImmutableList.copyOf(outputs));
    }

    @Override
    public void execute(ActionExecutionContext actionExecutionContext) {
    }

    @Override protected String computeKey() { return "action"; }
    @Override public ResourceSet estimateResourceConsumption(Executor executor) {
      return ResourceSet.ZERO;
    }
    @Override
    public String getMnemonic() {
      return "Null";
    }
  }

  /**
   * For a bunch of actions, gets the basenames of the paths and accumulates
   * them in a space separated string, like <code>foo.o bar.o baz.a</code>.
   */
  public static String baseNamesOf(Iterable<Artifact> artifacts) {
    List<String> baseNames = baseArtifactNames(artifacts);
    return Joiner.on(' ').join(baseNames);
  }

  /**
   * For a bunch of actions, gets the basenames of the paths, sorts them in alphabetical
   * order and accumulates them in a space separated string, for example
   * <code>bar.o baz.a foo.o</code>.
   */
  public static String sortedBaseNamesOf(Iterable<Artifact> artifacts) {
    List<String> baseNames = baseArtifactNames(artifacts);
    Collections.sort(baseNames);
    return Joiner.on(' ').join(baseNames);
  }

  /**
   * For a bunch of artifacts, gets the basenames and accumulates them in a
   * List.
   */
  public static List<String> baseArtifactNames(Iterable<Artifact> artifacts) {
    List<String> baseNames = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      baseNames.add(artifact.getExecPath().getBaseName());
    }
    return baseNames;
  }

  /**
   * For a bunch of artifacts, gets the exec paths and accumulates them in a
   * List.
   */
  public static List<String> execPaths(Iterable<Artifact> artifacts) {
    List<String> names = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      names.add(artifact.getExecPathString());
    }
    return names;
  }

  /**
   * For a bunch of artifacts, gets the pretty printed names and accumulates them in a List. Note
   * that this returns the root-relative paths, not the exec paths.
   */
  public static List<String> prettyArtifactNames(Iterable<Artifact> artifacts) {
    List<String> result = new ArrayList<>();
    for (Artifact artifact : artifacts) {
      result.add(artifact.prettyPrint());
    }
    return result;
  }

  public static List<String> prettyJarNames(Iterable<Artifact> jars) {
    List<String> result = new ArrayList<>();
    for (Artifact jar : jars) {
      result.add(jar.prettyPrint());
    }
    return result;
  }

  /**
   * Returns the closure of the predecessors of any of the given types, joining the basenames of the
   * artifacts into a space-separated string like "libfoo.a libbar.a libbaz.a".
   */
  public String predecessorClosureOf(Artifact artifact, FileType... types) {
    return predecessorClosureOf(Collections.singleton(artifact), types);
  }

  /**
   * Returns the closure of the predecessors of any of the given types.
   */
  public Collection<String> predecessorClosureAsCollection(Artifact artifact, FileType... types) {
    return predecessorClosureAsCollection(Collections.singleton(artifact), types);
  }

  /**
   * Returns the closure of the predecessors of any of the given types, joining the basenames of the
   * artifacts into a space-separated string like "libfoo.a libbar.a libbaz.a".
   */
  public String predecessorClosureOf(Iterable<Artifact> artifacts, FileType... types) {
    Set<Artifact> visited = artifactClosureOf(artifacts);
    return baseNamesOf(FileType.filter(visited, types));
  }

  /**
   * Returns the closure of the predecessors of any of the given types.
   */
  public Collection<String> predecessorClosureAsCollection(Iterable<Artifact> artifacts,
      FileType... types) {
    return baseArtifactNames(FileType.filter(artifactClosureOf(artifacts), types));
  }

  public String predecessorClosureOfJars(Iterable<Artifact> artifacts, FileType... types) {
    return baseNamesOf(FileType.filter(artifactClosureOf(artifacts), types));
  }

  public Collection<String> predecessorClosureJarsAsCollection(Iterable<Artifact> artifacts,
      FileType... types) {
    Set<Artifact> visited = artifactClosureOf(artifacts);
    return baseArtifactNames(FileType.filter(visited, types));
  }

  /**
   * Returns the closure over the input files of an action.
   */
  public Set<Artifact> inputClosureOf(Action action) {
    return artifactClosureOf(action.getInputs());
  }

  /**
   * Returns the closure over the input files of an artifact.
   */
  public Set<Artifact> artifactClosureOf(Artifact artifact) {
    return artifactClosureOf(Collections.singleton(artifact));
  }

  /**
   * Returns the closure over the input files of an artifact, filtered by the given matcher.
   */
  public Set<Artifact> filteredArtifactClosureOf(Artifact artifact, Predicate<Artifact> matcher) {
    return ImmutableSet.copyOf(Iterables.filter(artifactClosureOf(artifact), matcher));
  }

  /**
   * Returns the closure over the input files of a set of artifacts.
   */
  public Set<Artifact> artifactClosureOf(Iterable<Artifact> artifacts) {
    Set<Artifact> visited = new LinkedHashSet<>();
    List<Artifact> toVisit = Lists.newArrayList(artifacts);
    while (!toVisit.isEmpty()) {
      Artifact current = toVisit.remove(0);
      if (!visited.add(current)) {
        continue;
      }
      Action generatingAction = actionGraph.getGeneratingAction(current);
      if (generatingAction != null) {
        Iterables.addAll(toVisit, generatingAction.getInputs());
      }
    }
    return visited;
  }

  /**
   * Returns the closure over the input files of a set of artifacts, filtered by the given matcher.
   */
  public Set<Artifact> filteredArtifactClosureOf(Iterable<Artifact> artifacts,
      Predicate<Artifact> matcher) {
    return ImmutableSet.copyOf(Iterables.filter(artifactClosureOf(artifacts), matcher));
  }

  /**
   * Returns a predicate to match {@link Artifact}s with the given root-relative path suffix.
   */
  public static Predicate<Artifact> getArtifactSuffixMatcher(final String suffix) {
    return new Predicate<Artifact>() {
      @Override
      public boolean apply(Artifact input) {
        return input.getRootRelativePath().getPathString().endsWith(suffix);
      }
    };
  }

  /**
   * Finds all the actions that are instances of <code>actionClass</code>
   * in the transitive closure of prerequisites.
   */
  public <A extends Action> List<A> findTransitivePrerequisitesOf(Artifact artifact,
      Class<A> actionClass, Predicate<Artifact> allowedArtifacts) {
    List<A> actions = new ArrayList<>();
    Set<Artifact> visited = new LinkedHashSet<>();
    List<Artifact> toVisit = new LinkedList<>();
    toVisit.add(artifact);
    while (!toVisit.isEmpty()) {
      Artifact current = toVisit.remove(0);
      if (!visited.add(current)) {
        continue;
      }
      Action generatingAction = actionGraph.getGeneratingAction(current);
      if (generatingAction != null) {
        Iterables.addAll(toVisit, Iterables.filter(generatingAction.getInputs(), allowedArtifacts));
        if (actionClass.isInstance(generatingAction)) {
          actions.add(actionClass.cast(generatingAction));
        }
      }
    }
    return actions;
  }

  public <A extends Action> List<A> findTransitivePrerequisitesOf(
      Artifact artifact, Class<A> actionClass) {
    return findTransitivePrerequisitesOf(artifact, actionClass, Predicates.<Artifact>alwaysTrue());
  }

  /**
   * Looks in the given artifacts Iterable for the first Artifact whose path ends with the given
   * suffix and returns its generating Action.
   */
  public Action getActionForArtifactEndingWith(Iterable<Artifact> artifacts, String suffix) {
    Artifact a = getFirstArtifactEndingWith(artifacts, suffix);
    return a != null ? actionGraph.getGeneratingAction(a) : null;
  }

  /**
   * Looks in the given artifacts Iterable for the first Artifact whose path ends with the given
   * suffix and returns the Artifact.
   */
  public static Artifact getFirstArtifactEndingWith(
      Iterable<Artifact> artifacts, String suffix) {
    for (Artifact a : artifacts) {
      if (a.getExecPath().getPathString().endsWith(suffix)) {
        return a;
      }
    }
    return null;
  }

  /**
   * Returns the first artifact which is an input to "action" and has the
   * specified basename. An assertion error is raised if none is found.
   */
  public static Artifact getInput(Action action, String basename) {
    for (Artifact artifact : action.getInputs()) {
      if (artifact.getExecPath().getBaseName().equals(basename)) {
        return artifact;
      }
    }
    throw new AssertionError("No input with basename '" + basename + "' in action " + action);
  }

  /**
   * Returns true if an artifact that is an input to "action" with the specific
   * basename exists.
   */
  public static boolean hasInput(Action action, String basename) {
    try {
      getInput(action, basename);
      return true;
    } catch (AssertionError e) {
      return false;
    }
  }

  /**
   * Assert that an artifact is the primary output of its generating action.
   */
  public void assertPrimaryInputAndOutputArtifacts(Artifact input, Artifact output) {
    Action generatingAction = actionGraph.getGeneratingAction(output);
    assertThat(generatingAction).isNotNull();
    assertThat(generatingAction.getPrimaryOutput()).isEqualTo(output);
    assertThat(generatingAction.getPrimaryInput()).isEqualTo(input);
  }

  /**
   * Returns the first artifact which is an output of "action" and has the
   * specified basename. An assertion error is raised if none is found.
   */
  public static Artifact getOutput(Action action, String basename) {
    for (Artifact artifact : action.getOutputs()) {
      if (artifact.getExecPath().getBaseName().equals(basename)) {
        return artifact;
      }
    }
    throw new AssertionError("No output with basename '" + basename + "' in action " + action);
  }

  public static void registerActionWith(Action action, MutableActionGraph actionGraph) {
    try {
      actionGraph.registerAction(action);
    } catch (ActionConflictException e) {
      throw new UncheckedActionConflictException(e);
    }
  }
}
