package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import javax.annotation.Nullable;

interface RewoundActionSynchronizer {
  SilentCloseable lockForRewoundActionBeforePreparation(Action action) throws InterruptedException;

  SilentCloseable lockForAnyActionBeforeExecution(Action action) throws InterruptedException;

  RewoundActionSynchronizer NOOP =
      new RewoundActionSynchronizer() {
        @Override
        public SilentCloseable lockForRewoundActionBeforePreparation(Action action) {
          return () -> {};
        }

        @Override
        public SilentCloseable lockForAnyActionBeforeExecution(Action action) {
          return () -> {};
        }
      };

  static RewoundActionSynchronizer create(
      boolean rewindingEnabled, ActionFileSystemType actionFileSystemType) {
    if (rewindingEnabled && actionFileSystemType.shouldDoEagerActionPrep()) {
      return new BazelRewoundActionSynchronizer();
    } else {
      return NOOP;
    }
  }

  final class BazelRewoundActionSynchronizer implements RewoundActionSynchronizer {
    @Nullable private volatile ReentrantReadWriteLock coarseLock = new ReentrantReadWriteLock();
    @Nullable private volatile ConcurrentHashMap<Artifact, ReentrantReadWriteLock> fineLocks = null;

    private BazelRewoundActionSynchronizer() {}

    @Override
    public SilentCloseable lockForRewoundActionBeforePreparation(Action action)
        throws InterruptedException {
      if (coarseLock != null) {
        coarseLock.writeLock().lockInterruptibly();
        coarseLock = null;
      }
      return null;
    }

    @Override
    public SilentCloseable lockForAnyActionBeforeExecution(Action action)
        throws InterruptedException {
      return null;
    }
  }
}
