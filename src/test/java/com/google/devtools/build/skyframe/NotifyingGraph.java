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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Maps;
import com.google.common.collect.Maps.EntryTransformer;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.GroupedList;

import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Class that allows clients to be notified on each access of the graph. Clients can simply track
 * accesses, or they can block to achieve desired synchronization. Clients should call
 * {@link TrackingAwaiter#INSTANCE#assertNoErrors} at the end of tests in case exceptions were
 * swallowed in async threads.
 *
 * <p>While this class nominally always implements a {@link ProcessableGraph}, it will throw if any
 * of the methods in {@link ProcessableGraph} that are not in {@link ThinNodeQueryableGraph} are
 * called on it and its {@link #delegate} is not a {@link ProcessableGraph}. This lack of type
 * safety is so that a {@code NotifyingGraph} can be returned by {@link #makeNotifyingTransformer}
 * and used in {@link MemoizingEvaluator#injectGraphTransformerForTesting}.
 */
public class NotifyingGraph<TGraph extends ThinNodeQueryableGraph> implements ProcessableGraph {
  public static Function<ThinNodeQueryableGraph, ProcessableGraph> makeNotifyingTransformer(
      final Listener listener) {
    return new Function<ThinNodeQueryableGraph, ProcessableGraph>() {
      @Nullable
      @Override
      public ProcessableGraph apply(ThinNodeQueryableGraph queryableGraph) {
        if (queryableGraph instanceof InMemoryGraph) {
          return new NotifyingInMemoryGraph((InMemoryGraph) queryableGraph, listener);
        } else {
          return new NotifyingGraph<>(queryableGraph, listener);
        }
      }
    };
  }

  protected final TGraph delegate;
  private final Listener graphListener;

  private final EntryTransformer<SkyKey, ThinNodeEntry, NodeEntry> wrapEntry =
      new EntryTransformer<SkyKey, ThinNodeEntry, NodeEntry>() {
        @Nullable
        @Override
        public NotifyingNodeEntry transformEntry(SkyKey key, @Nullable ThinNodeEntry nodeEntry) {
          return wrapEntry(key, nodeEntry);
        }
      };

  NotifyingGraph(TGraph delegate, Listener graphListener) {
    this.delegate = delegate;
    this.graphListener = new ErrorRecordingDelegatingListener(graphListener);
  }

  private ProcessableGraph getProcessableDelegate() {
    return (ProcessableGraph) delegate;
  }

  @Override
  public void remove(SkyKey key) {
    getProcessableDelegate().remove(key);
  }

  @Override
  public Map<SkyKey, NodeEntry> createIfAbsentBatch(Iterable<SkyKey> keys) {
    for (SkyKey key : keys) {
      graphListener.accept(key, EventType.CREATE_IF_ABSENT, Order.BEFORE, null);
    }
    return Maps.transformEntries(getProcessableDelegate().createIfAbsentBatch(keys), wrapEntry);
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatch(Iterable<SkyKey> keys) {
    if (delegate instanceof ProcessableGraph) {
      return Maps.transformEntries(getProcessableDelegate().getBatch(keys), wrapEntry);
    } else {
      return Maps.transformEntries(delegate.getBatch(keys), wrapEntry);
    }
  }

  @Nullable
  @Override
  public NodeEntry get(SkyKey key) {
    return wrapEntry(key, getProcessableDelegate().get(key));
  }

  /** Subclasses should override if they wish to subclass NotifyingNodeEntry. */
  @Nullable
  protected NotifyingNodeEntry wrapEntry(SkyKey key, @Nullable ThinNodeEntry entry) {
    return entry == null ? null : new NotifyingNodeEntry(key, entry);
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
  }
}
