package com.google.devtools.build.lib.bazel;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.blaze.BlazeModule;
import com.google.devtools.build.lib.blaze.BlazeRuntime;
import com.google.devtools.build.lib.blaze.Command;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.LocalDiffAwareness;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Provides the {@link DiffAwareness} implementation that uses the Java watch service.
 */
public class BazelDiffAwarenessModule extends BlazeModule {

  private final AtomicReference<EventBus> eventBusRef = new AtomicReference<>();

  @Override
  public void beforeCommand(BlazeRuntime blazeRuntime, Command command) {
    eventBusRef.set(blazeRuntime.getEventBus());
  }

  @Override
  public Iterable<DiffAwareness.Factory> getDiffAwarenessFactories(boolean watchFS) {
    Builder<DiffAwareness.Factory> builder = ImmutableList.builder();
    if (watchFS) {
      builder.add(new LocalDiffAwareness.Factory());
    }
    return builder.build();
  }
}
