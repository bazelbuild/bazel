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
package com.google.devtools.build.skyframe;

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.collect.Maps;
import com.google.common.collect.Maps.EntryTransformer;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.GroupedList;

import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Class that allows clients to be notified on each access of the graph. Clients can simply track
 * accesses, or they can block to achieve desired synchronization. Clients should call {@link
 * TrackingAwaiter#INSTANCE#assertNoErrors} at the end of tests in case exceptions were swallowed in
 * async threads.
 */
public class NotifyingHelper {
  public static MemoizingEvaluator.GraphTransformerForTesting makeNotifyingTransformer(
      final Listener listener) {
    return new MemoizingEvaluator.GraphTransformerForTesting() {
      @Override
      public InMemoryGraph transform(InMemoryGraph graph) {
        return new NotifyingInMemoryGraph(graph, listener);
      }

      @Override
      public InvalidatableGraph transform(InvalidatableGraph graph) {
        return new NotifyingInvalidatableGraph(graph, listener);
      }

      @Override
      public ProcessableGraph transform(ProcessableGraph graph) {
        return new NotifyingProcessableGraph(graph, listener);
      }
    };
  }

  protected final Listener graphListener;

  protected final EntryTransformer<SkyKey, ThinNodeEntry, NodeEntry> wrapEntry =
      new EntryTransformer<SkyKey, ThinNodeEntry, NodeEntry>() {
        @Nullable
        @Override
        public NotifyingNodeEntry transformEntry(SkyKey key, @Nullable ThinNodeEntry nodeEntry) {
          return wrapEntry(key, nodeEntry);
        }
      };

  NotifyingHelper(Listener graphListener) {
    this.graphListener = new ErrorRecordingDelegatingListener(graphListener);
  }

  /** Subclasses should override if they wish to subclass NotifyingNodeEntry. */
  @Nullable
  protected NotifyingNodeEntry wrapEntry(SkyKey key, @Nullable ThinNodeEntry entry) {
    return entry == null ? null : new NotifyingNodeEntry(key, entry);
  }

  static class NotifyingInvalidatableGraph implements InvalidatableGraph {
    private final InvalidatableGraph delegate;
    private final NotifyingHelper notifyingHelper;

    NotifyingInvalidatableGraph(InvalidatableGraph delegate, Listener graphListener) {
      this.notifyingHelper = new NotifyingHelper(graphListener);
      this.delegate = delegate;
    }

    NotifyingInvalidatableGraph(InvalidatableGraph delegate, NotifyingHelper helper) {
      this.notifyingHelper = helper;
      this.delegate = delegate;
    }

    @Override
    public Map<SkyKey, NodeEntry> getBatch(Iterable<SkyKey> keys) {
      return Maps.transformEntries(delegate.getBatch(keys), notifyingHelper.wrapEntry);
    }
  }

  static class NotifyingProcessableGraph implements ProcessableGraph {
    protected final ProcessableGraph delegate;
    protected final NotifyingHelper notifyingHelper;

    NotifyingProcessableGraph(ProcessableGraph delegate, Listener graphListener) {
      this.notifyingHelper = new NotifyingHelper(graphListener);
      this.delegate = delegate;
    }

    NotifyingProcessableGraph(ProcessableGraph delegate, NotifyingHelper helper) {
      this.notifyingHelper = helper;
      this.delegate = delegate;
    }

    @Override
    public void remove(SkyKey key) {
      delegate.remove(key);
    }

    @Override
    public Map<SkyKey, NodeEntry> createIfAbsentBatch(Iterable<SkyKey> keys) {
      for (SkyKey key : keys) {
        notifyingHelper.graphListener.accept(key, EventType.CREATE_IF_ABSENT, Order.BEFORE, null);
      }
      return Maps.transformEntries(delegate.createIfAbsentBatch(keys), notifyingHelper.wrapEntry);
    }

    @Override
    public Map<SkyKey, NodeEntry> getBatchWithFieldHints(
        Iterable<SkyKey> keys, EnumSet<NodeEntryField> fields) {
      return Maps.transformEntries(
          delegate.getBatchWithFieldHints(keys, fields), notifyingHelper.wrapEntry);
    }

    @Nullable
    @Override
    public NodeEntry get(SkyKey key) {
      return notifyingHelper.wrapEntry(key, delegate.get(key));
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
    GET_TEMPORARY_DIRECT_DEPS,
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

  /**
   * Whether the given event is about to happen or has just happened. For some events, both will be
   * published, for others, only one. When writing tests, if you need an additional one to be
   * published, feel free to add it.
   */
  public enum Order {
    BEFORE,
    AFTER
  }

  /** Receiver to be informed when an event for a given key occurs. */
  public interface Listener {
    @ThreadSafe
    void accept(SkyKey key, EventType type, Order order, Object context);

    Listener NULL_LISTENER =
        new Listener() {
          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {}
        };
  }

  private static class ErrorRecordingDelegatingListener implements Listener {
    private final Listener delegate;

    private ErrorRecordingDelegatingListener(Listener delegate) {
      this.delegate = delegate;
    }

    @Override
    public void accept(SkyKey key, EventType type, Order order, Object context) {
      try {
        delegate.accept(key, type, order, context);
      } catch (Exception e) {
        TrackingAwaiter.INSTANCE.injectExceptionAndMessage(
            e, "In NotifyingGraph: " + Joiner.on(", ").join(key, type, order, context));
        throw e;
      }
    }
  }

  /** {@link NodeEntry} that informs a {@link Listener} of various method calls. */
  protected class NotifyingNodeEntry extends DelegatingNodeEntry {
    private final SkyKey myKey;
    private final ThinNodeEntry delegate;

    protected NotifyingNodeEntry(SkyKey key, ThinNodeEntry delegate) {
      myKey = key;
      this.delegate = delegate;
    }

    @Override
    protected NodeEntry getDelegate() {
      return (NodeEntry) delegate;
    }

    @Override
    protected ThinNodeEntry getThinDelegate() {
      return delegate;
    }

    @Override
    public DependencyState addReverseDepAndCheckIfDone(SkyKey reverseDep) {
      graphListener.accept(myKey, EventType.ADD_REVERSE_DEP, Order.BEFORE, reverseDep);
      DependencyState result = super.addReverseDepAndCheckIfDone(reverseDep);
      graphListener.accept(myKey, EventType.ADD_REVERSE_DEP, Order.AFTER, reverseDep);
      return result;
    }

    @Override
    public void removeReverseDep(SkyKey reverseDep) {
      graphListener.accept(myKey, EventType.REMOVE_REVERSE_DEP, Order.BEFORE, reverseDep);
      super.removeReverseDep(reverseDep);
      graphListener.accept(myKey, EventType.REMOVE_REVERSE_DEP, Order.AFTER, reverseDep);
    }

    @Override
    public GroupedList<SkyKey> getTemporaryDirectDeps() {
      graphListener.accept(myKey, EventType.GET_TEMPORARY_DIRECT_DEPS, Order.BEFORE, null);
      return super.getTemporaryDirectDeps();
    }

    @Override
    public boolean signalDep(Version childVersion) {
      graphListener.accept(myKey, EventType.SIGNAL, Order.BEFORE, childVersion);
      boolean result = super.signalDep(childVersion);
      graphListener.accept(myKey, EventType.SIGNAL, Order.AFTER, childVersion);
      return result;
    }

    @Override
    public Set<SkyKey> setValue(SkyValue value, Version version) {
      graphListener.accept(myKey, EventType.SET_VALUE, Order.BEFORE, value);
      Set<SkyKey> result = super.setValue(value, version);
      graphListener.accept(myKey, EventType.SET_VALUE, Order.AFTER, value);
      return result;
    }

    @Override
    public MarkedDirtyResult markDirty(boolean isChanged) {
      graphListener.accept(myKey, EventType.MARK_DIRTY, Order.BEFORE, isChanged);
      MarkedDirtyResult result = super.markDirty(isChanged);
      graphListener.accept(myKey, EventType.MARK_DIRTY, Order.AFTER, isChanged);
      return result;
    }

    @Override
    public Set<SkyKey> markClean() {
      graphListener.accept(myKey, EventType.MARK_CLEAN, Order.BEFORE, this);
      Set<SkyKey> result = super.markClean();
      graphListener.accept(myKey, EventType.MARK_CLEAN, Order.AFTER, this);
      return result;
    }

    @Override
    public boolean isChanged() {
      graphListener.accept(myKey, EventType.IS_CHANGED, Order.BEFORE, this);
      return super.isChanged();
    }

    @Override
    public boolean isDirty() {
      graphListener.accept(myKey, EventType.IS_DIRTY, Order.BEFORE, this);
      return super.isDirty();
    }

    @Override
    public boolean isReady() {
      graphListener.accept(myKey, EventType.IS_READY, Order.BEFORE, this);
      return super.isReady();
    }

    @Override
    public SkyValue getValueMaybeWithMetadata() {
      graphListener.accept(myKey, EventType.GET_VALUE_WITH_METADATA, Order.BEFORE, this);
      return super.getValueMaybeWithMetadata();
    }

    @Override
    public DependencyState checkIfDoneForDirtyReverseDep(SkyKey reverseDep) {
      graphListener.accept(myKey, EventType.CHECK_IF_DONE, Order.BEFORE, reverseDep);
      return super.checkIfDoneForDirtyReverseDep(reverseDep);
    }

    @Override
    public Iterable<SkyKey> getAllDirectDepsForIncompleteNode() {
      graphListener.accept(
          myKey, EventType.GET_ALL_DIRECT_DEPS_FOR_INCOMPLETE_NODE, Order.BEFORE, this);
      return super.getAllDirectDepsForIncompleteNode();
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("delegate", getThinDelegate()).toString();
    }
  }
}
