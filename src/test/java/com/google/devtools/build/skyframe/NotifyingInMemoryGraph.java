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
package com.google.devtools.build.skyframe;

import com.google.common.base.Joiner;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.util.ArrayList;
import java.util.Set;

/**
 * Class that allows clients to be notified on each access of the graph. Clients can simply track
 * accesses, or they can block to achieve desired synchronization. Clients should call
 * {@link #assertNoExceptions} at the end of tests in case exceptions were swallowed in async
 * threads.
 */
public class NotifyingInMemoryGraph extends InMemoryGraph {
  private final Listener graphListener;
  private final ArrayList<Exception> unexpectedExceptions = new ArrayList<>();

  public NotifyingInMemoryGraph(Listener graphListener) {
    this.graphListener = new ErrorRecordingDelegatingListener(graphListener);
  }

  protected NodeEntry createIfAbsent(SkyKey key) {
    graphListener.accept(key, EventType.CREATE_IF_ABSENT, Order.BEFORE, null);
    NodeEntry newval = getEntry(key);
    NodeEntry oldval = getNodeMap().putIfAbsent(key, newval);
    return oldval == null ? newval : oldval;
  }

  /**
   * Should be called at end of test (ideally in an {@code @After} method) to assert that no
   * exceptions were thrown during calls to the listener.
   */
  public void assertNoExceptions() {
    Truth.assertThat(unexpectedExceptions).isEmpty();
  }

  // Subclasses should override if they wish to subclass NotifyingNodeEntry.
  protected NotifyingNodeEntry getEntry(SkyKey key) {
    return new NotifyingNodeEntry(key);
  }

  /** Receiver to be informed when an event for a given key occurs. */
  public interface Listener {
    @ThreadSafe
    void accept(SkyKey key, EventType type, Order order, Object context);

    public static Listener NULL_LISTENER = new Listener() {
      @Override
      public void accept(SkyKey key, EventType type, Order order, Object context) {}
    };
  }

  private class ErrorRecordingDelegatingListener implements Listener {
    private final Listener delegate;

    private ErrorRecordingDelegatingListener(Listener delegate) {
      this.delegate = delegate;
    }

    @Override
    public void accept(SkyKey key, EventType type, Order order, Object context) {
      try {
        delegate.accept(key, type, order, context);
      } catch (Exception e) {
        TrackingAwaiter.INSTANCE.injectExceptionAndMessage(e,
            "In NotifyingInMemoryGraph: " + Joiner.on(", ").join(key, type, order, context));
        throw e;
      }
    }
  }
  /**
   * Graph/value entry events that the receiver can be informed of. When writing tests, feel free to
   * add additional events here if needed.
   */
  public enum EventType {
    CREATE_IF_ABSENT,
    ADD_REVERSE_DEP,
    REMOVE_REVERSE_DEP,
    SIGNAL,
    SET_VALUE,
    MARK_DIRTY,
    MARK_CLEAN,
    IS_CHANGED,
    GET_VALUE_WITH_METADATA,
    IS_DIRTY,
    IS_READY,
    CHECK_IF_DONE,
    GET_ALL_DIRECT_DEPS_FOR_INCOMPLETE_NODE
  }

  public enum Order {
    BEFORE,
    AFTER
  }

  /**
   * Note that the methods in this class intentionally do not have the {@code synchronized}
   * keyword! Each of them invokes the synchronized method on {@link InMemoryNodeEntry} it
   * overrides, which provides the required synchronization for state owned by that base class.
   *
   * <p>These methods are not synchronized because several test cases control the flow of
   * execution by blocking until notified by the callbacks executed in these methods. If these
   * overrides were synchronized, they wouldn't get the chance to execute these callbacks.
   */
  protected class NotifyingNodeEntry extends InMemoryNodeEntry {
    private final SkyKey myKey;

    protected NotifyingNodeEntry(SkyKey key) {
      myKey = key;
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public DependencyState addReverseDepAndCheckIfDone(SkyKey reverseDep) {
      graphListener.accept(myKey, EventType.ADD_REVERSE_DEP, Order.BEFORE, reverseDep);
      DependencyState result = super.addReverseDepAndCheckIfDone(reverseDep);
      graphListener.accept(myKey, EventType.ADD_REVERSE_DEP, Order.AFTER, reverseDep);
      return result;
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public void removeReverseDep(SkyKey reverseDep) {
      graphListener.accept(myKey, EventType.REMOVE_REVERSE_DEP, Order.BEFORE, reverseDep);
      super.removeReverseDep(reverseDep);
      graphListener.accept(myKey, EventType.REMOVE_REVERSE_DEP, Order.AFTER, reverseDep);
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public boolean signalDep(Version childVersion) {
      graphListener.accept(myKey, EventType.SIGNAL, Order.BEFORE, childVersion);
      boolean result = super.signalDep(childVersion);
      graphListener.accept(myKey, EventType.SIGNAL, Order.AFTER, childVersion);
      return result;
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public Set<SkyKey> setValue(SkyValue value, Version version) {
      graphListener.accept(myKey, EventType.SET_VALUE, Order.BEFORE, value);
      Set<SkyKey> result = super.setValue(value, version);
      graphListener.accept(myKey, EventType.SET_VALUE, Order.AFTER, value);
      return result;
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public MarkedDirtyResult markDirty(boolean isChanged) {
      graphListener.accept(myKey, EventType.MARK_DIRTY, Order.BEFORE, isChanged);
      MarkedDirtyResult result = super.markDirty(isChanged);
      graphListener.accept(myKey, EventType.MARK_DIRTY, Order.AFTER, isChanged);
      return result;
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public Set<SkyKey> markClean() {
      graphListener.accept(myKey, EventType.MARK_CLEAN, Order.BEFORE, this);
      Set<SkyKey> result = super.markClean();
      graphListener.accept(myKey, EventType.MARK_CLEAN, Order.AFTER, this);
      return result;
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public boolean isChanged() {
      graphListener.accept(myKey, EventType.IS_CHANGED, Order.BEFORE, this);
      return super.isChanged();
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public boolean isDirty() {
      graphListener.accept(myKey, EventType.IS_DIRTY, Order.BEFORE, this);
      return super.isDirty();
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public boolean isReady() {
      graphListener.accept(myKey, EventType.IS_READY, Order.BEFORE, this);
      return super.isReady();
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public SkyValue getValueMaybeWithMetadata() {
      graphListener.accept(myKey, EventType.GET_VALUE_WITH_METADATA, Order.BEFORE, this);
      return super.getValueMaybeWithMetadata();
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep) {
      graphListener.accept(myKey, EventType.CHECK_IF_DONE, Order.BEFORE, reverseDep);
      return super.checkIfDoneForDirtyReverseDep(reverseDep);
    }

    @SuppressWarnings("UnsynchronizedOverridesSynchronized") // See the class doc for details.
    @Override
    public Iterable<SkyKey> getAllDirectDepsForIncompleteNode() {
      graphListener.accept(
          myKey, EventType.GET_ALL_DIRECT_DEPS_FOR_INCOMPLETE_NODE, Order.BEFORE, this);
      return super.getAllDirectDepsForIncompleteNode();
    }
  }
}
