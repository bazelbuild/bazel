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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimaps;
import com.google.devtools.build.lib.actions.ActionContext.ActionContextRegistry;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Test utility that allows for controlling the behavior of spawns by using {@link SpawnShim}.
 *
 * <p>To install in integration tests, use {@link ControllableActionStrategyModule}.
 */
public final class SpawnController {

  /** The means of controlling {@link SpawnStrategy#exec} calls. */
  public interface SpawnShim {
    ExecResult getExecResult(Spawn spawn, ActionExecutionContext context)
        throws IOException, InterruptedException;
  }

  /**
   * Represents the desired behavior of a {@link SpawnStrategy#exec} call. Instances represent one
   * of the following:
   *
   * <ul>
   *   <li>A {@link SpawnResult}, created with {@link #of}.
   *   <li>An {@link ExecException}, created with {@link #ofException}.
   *   <li>A delegate to the underlying {@link SpawnStrategy}, created with {@link #delegate}.
   * </ul>
   */
  public static final class ExecResult {
    private static final ExecResult DELEGATE =
        new ExecResult(/*execException=*/ null, /*spawnResult=*/ null);

    @Nullable private final ExecException execException;
    @Nullable private final SpawnResult spawnResult;

    private ExecResult(@Nullable ExecException execException, @Nullable SpawnResult spawnResult) {
      this.execException = execException;
      this.spawnResult = spawnResult;
    }

    /** Override the action by returning the provided {@link SpawnResult}. */
    public static ExecResult of(SpawnResult spawnResult) {
      return new ExecResult(/*execException=*/ null, checkNotNull(spawnResult));
    }

    /** Override the action by throwing the provided {@link ExecException}. */
    public static ExecResult ofException(ExecException execException) {
      return new ExecResult(checkNotNull(execException), /*spawnResult=*/ null);
    }

    /** Do not override the action. Allow the underlying {@link SpawnStrategy} to handle it. */
    public static ExecResult delegate() {
      return DELEGATE;
    }
  }

  private final List<String> executedSpawnDescriptions =
      Collections.synchronizedList(new ArrayList<>());

  private final ListMultimap<String, SpawnShim> spawnShims =
      Multimaps.synchronizedListMultimap(LinkedListMultimap.create());

  /**
   * Returns a list of all executed spawn descriptions seen by strategies created via {@link #wrap}
   * (in order) since the last call to {@link #clearExecutedSpawnDescriptions}.
   */
  public ImmutableList<String> getExecutedSpawnDescriptions() {
    return ImmutableList.copyOf(executedSpawnDescriptions);
  }

  /** Clears the list of executed spawn descriptions. */
  public void clearExecutedSpawnDescriptions() {
    executedSpawnDescriptions.clear();
  }

  /**
   * Injects custom spawn behavior for {@linkplain #wrap controllable strategies}.
   *
   * <p>The given {@link SpawnShim} is enqueued for a single execution of a spawn with the given
   * description. When a matching spawn is seen, an associated {@link SpawnShim} is dequeued and
   * used in a FIFO manner. If there are no matching shims enqueued, the delegate strategy is used.
   */
  public void addSpawnShim(String spawnDescription, SpawnShim spawnShim) {
    spawnShims.put(spawnDescription, spawnShim);
  }

  /**
   * Creates a new {@link SpawnStrategy} that picks up custom behavior added via {@link
   * #addSpawnShim} and delegates to the given {@code delegate} if necessary.
   */
  SpawnStrategy wrap(SpawnStrategy delegate) {
    return new ControllableSpawnStrategy(delegate);
  }

  /**
   * Checks that all spawn shims added via {@link #addSpawnShim} have been consumed, throwing an
   * {@link IllegalStateException} if any remain.
   *
   * <p>This can be used to verify that shims were configured correctly.
   */
  public void verifyAllShimsConsumed() {
    checkState(spawnShims.isEmpty(), "Remaining spawn shims: %s", spawnShims);
  }

  private final class ControllableSpawnStrategy implements SpawnStrategy {
    private final SpawnStrategy delegate;

    ControllableSpawnStrategy(SpawnStrategy delegate) {
      this.delegate = checkNotNull(delegate);
    }

    @Override
    public ImmutableList<SpawnResult> exec(
        Spawn spawn, ActionExecutionContext actionExecutionContext)
        throws ExecException, InterruptedException {
      String description = spawn.getResourceOwner().describe();
      executedSpawnDescriptions.add(description);

      List<SpawnShim> events = spawnShims.get(description);
      if (!events.isEmpty()) {
        ExecResult execResult;
        try {
          execResult = events.remove(0).getExecResult(spawn, actionExecutionContext);
        } catch (IOException e) {
          throw new SpawnShimException(e, description);
        }
        if (execResult.execException != null) {
          throw execResult.execException;
        }
        if (execResult.spawnResult != null) {
          return ImmutableList.of(execResult.spawnResult);
        }
      }

      return delegate.exec(spawn, actionExecutionContext);
    }

    @Override
    public boolean canExec(Spawn spawn, ActionContextRegistry actionContextRegistry) {
      return delegate.canExec(spawn, actionContextRegistry);
    }
  }

  private static final class SpawnShimException extends ExecException {
    private final String description;

    SpawnShimException(IOException e, String description) {
      super(e);
      this.description = description;
    }

    @Override
    protected String getMessageForActionExecutionException() {
      return "In a test shim, failed to determine ExecException for action: "
          + description
          + ": "
          + getCause().getMessage();
    }

    @Override
    protected FailureDetail getFailureDetail(String message) {
      return CrashFailureDetails.forThrowable(this);
    }
  }
}
