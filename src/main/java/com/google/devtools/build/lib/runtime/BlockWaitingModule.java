package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;

/** A {@link BlazeModule} that await for submitted tasks to terminate after every command. */
public class BlockWaitingModule extends BlazeModule {
  private @Nullable ExecutorService executorService;

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    checkState(executorService == null, "executorService must be null");

    executorService =
        Executors.newCachedThreadPool(
            new ThreadFactoryBuilder().setNameFormat("block-waiting-%d").build());
  }

  public void submit(Runnable task) {
    checkNotNull(executorService, "executorService must not be null");

    executorService.submit(task);
  }

  @Override
  public void afterCommand() throws AbruptExitException {
    checkNotNull(executorService, "executorService must not be null");

    executorService.shutdown();
    try {
      executorService.awaitTermination(Long.MAX_VALUE, SECONDS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }

    executorService = null;
  }
}
