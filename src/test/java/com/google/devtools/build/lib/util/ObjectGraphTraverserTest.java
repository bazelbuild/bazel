// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.refEq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.collect.ConcurrentIdentitySet;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.DomainSpecificTraverser;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.EdgeType;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.FieldCache;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.Traversal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ObjectGraphTraverserTest {
  private static final class Edge {
    private final Object from;
    private final Object to;
    private final EdgeType type;

    private Edge(Object from, Object to, EdgeType type) {
      this.from = from;
      this.to = to;
      this.type = type;
    }

    private static Edge of(Object from, Object to, EdgeType type) {
      return new Edge(from, to, type);
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof Edge)) {
        return false;
      }

      Edge that = (Edge) o;
      return that.from == from && that.to == to && that.type == type;
    }

    @Override
    public int hashCode() {
      return Objects.hash(System.identityHashCode(from), System.identityHashCode(to), type);
    }
  }

  private static final class LoggingObjectReceiver implements ObjectGraphTraverser.ObjectReceiver {
    private List<Object> objects = new ArrayList<>();
    private Map<Object, String> objectContexts = new HashMap<>();
    private List<Edge> edges = new ArrayList<>();
    private Map<Edge, String> edgeContexts = new HashMap<>();

    @Override
    public void objectFound(Object o, String context) {
      objects.add(o);
      if (context != null) {
        objectContexts.put(o, context);
      }
    }

    @Override
    public void edgeFound(Object from, Object to, String toContext, EdgeType edgeType) {
      Edge edge = Edge.of(from, to, edgeType);

      edges.add(edge);
      if (toContext != null) {
        edgeContexts.put(edge, toContext);
      }
    }
  }

  private ObjectGraphTraverser createObjectGraphTraverser(
      DomainSpecificTraverser domainSpecific,
      ConcurrentIdentitySet seen,
      LoggingObjectReceiver receiver,
      boolean collectContext) {
    ImmutableList<DomainSpecificTraverser> traversers =
        domainSpecific == null ? ImmutableList.of() : ImmutableList.of(domainSpecific);
    return new ObjectGraphTraverser(
        new FieldCache(traversers), seen, collectContext, receiver, null);
  }

  @Test
  public void smoke() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object array = new Object[] {o2};
    Object pair = Pair.of(o1, array);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);
    cut.traverse(pair);

    assertThat(receiver.objects).containsExactly(o1, o2, array, pair);
    assertThat(receiver.edges).hasSize(3);
  }

  @Test
  public void testAdmit() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object pair1 = Pair.of(o1, o1);
    Object pair2 = Pair.of(o2, o2);
    Object pair3 = Pair.of(pair1, pair2);

    DomainSpecificTraverser domainSpecific = mock(DomainSpecificTraverser.class);
    when(domainSpecific.admit(any())).thenAnswer(i -> i.getArgument(0) != pair2);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(domainSpecific, seen, receiver, false);
    cut.traverse(pair3);

    assertThat(receiver.objects).containsExactly(o1, pair1, pair3);
    assertThat(receiver.edges).hasSize(3);
  }

  @Test
  public void testCustomTraversal() {
    Object o1 = new Object();
    Object o2 = new Object();

    DomainSpecificTraverser domainSpecific = mock(DomainSpecificTraverser.class);
    when(domainSpecific.admit(any())).thenReturn(true);
    when(domainSpecific.maybeTraverse(any(), any()))
        .thenAnswer(
            i -> {
              Object arg = i.getArgument(0);
              Traversal traversal = i.getArgument(1);

              if (arg != o1) {
                return false;
              }

              traversal.objectFound(o1, null);
              traversal.edgeFound(o2, null);
              return true;
            });

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(domainSpecific, seen, receiver, false);
    cut.traverse(o1);

    assertThat(receiver.objects).containsExactly(o1, o2);
    assertThat(receiver.edges).containsExactly(Edge.of(o1, o2, EdgeType.CURRENT_TRAVERSAL));
  }

  @Test
  public void testIgnoredFields() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object pair = Pair.of(o1, o2);

    DomainSpecificTraverser domainSpecific = mock(DomainSpecificTraverser.class);
    when(domainSpecific.ignoredFields(Pair.class)).thenReturn(ImmutableSet.of("second"));
    when(domainSpecific.admit(any())).thenReturn(true);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(domainSpecific, seen, receiver, false);
    cut.traverse(pair);

    assertThat(receiver.objects).containsExactly(o1, pair);
    assertThat(receiver.edges).containsExactly(Edge.of(pair, o1, EdgeType.CURRENT_TRAVERSAL));
  }

  @Test
  public void testSeenObjects() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object pair = Pair.of(o1, o2);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    var unused = seen.add(o2);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);
    cut.traverse(pair);

    assertThat(receiver.objects).containsExactly(o1, pair);
    assertThat(receiver.edges)
        .containsExactly(
            Edge.of(pair, o1, EdgeType.CURRENT_TRAVERSAL),
            Edge.of(pair, o2, EdgeType.ALREADY_SEEN));
  }

  private static final class Outer {
    private Inner createInner() {
      return new Inner();
    }

    private class Inner {
      // Java is clever and will optimize out the reference to Outer without this
      @SuppressWarnings("unused")
      private Outer getOuter() {
        return Outer.this;
      }
    }
  }

  @Test
  public void testNonStaticClassTraversesEnclosingClass() {
    Outer outer = new Outer();
    Outer.Inner inner = outer.createInner();

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);

    cut.traverse(inner);
    assertThat(receiver.objects).containsExactly(outer, inner);
  }

  @Test
  public void testLambdaClosingOverNothingReported() {
    Object o1 = new Object();
    Supplier<Object> lambda = () -> 3;
    Object pair = Pair.of(o1, lambda);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);

    cut.traverse(pair);
    assertThat(receiver.objects).containsExactly(pair, o1, lambda);
  }

  @Test
  public void testLambdaClosingOverNothingReportedWhenReferencedTwice() {
    Supplier<Object> lambda = () -> 3;
    Object pair = Pair.of(lambda, lambda);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);

    cut.traverse(pair);
    assertThat(receiver.objects).containsExactly(pair, lambda);
  }

  @Test
  public void testValuesClosedOverReported() {
    Object o1 = new Object();
    Supplier<Object> lambda = () -> o1;

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);

    cut.traverse(lambda);
    assertThat(receiver.objects).containsExactly(lambda, o1);
  }

  @Test
  public void testMultipleClosuresWithSameCodeReported() {
    Object o1 = new Object();
    Object o2 = new Object();
    Function<Object, Supplier<Object>> generator = o -> () -> o;
    Object l1 = generator.apply(o1);
    Object l2 = generator.apply(o2);
    Object pair = Pair.of(l1, l2);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    ObjectGraphTraverser cut = createObjectGraphTraverser(null, seen, receiver, false);

    cut.traverse(pair);
    assertThat(receiver.objects).containsExactly(pair, l1, l2, o1, o2);
  }

  @Test
  public void testEdgeContexts() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object array = new Object[] {o2};
    Object pair = Pair.of(o1, array);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    DomainSpecificTraverser domainSpecific = mock(DomainSpecificTraverser.class);
    when(domainSpecific.admit(any())).thenReturn(true);
    when(domainSpecific.contextForField(refEq(pair), any(), any(), refEq(o1)))
        .thenReturn("o1context");
    when(domainSpecific.contextForArrayItem(refEq(array), any(), refEq(o2)))
        .thenReturn("o2context");
    ObjectGraphTraverser cut = createObjectGraphTraverser(domainSpecific, seen, receiver, true);

    cut.traverse(pair);
    assertThat(receiver.edgeContexts)
        .containsEntry(Edge.of(pair, o1, EdgeType.CURRENT_TRAVERSAL), "o1context");
    assertThat(receiver.edgeContexts)
        .containsEntry(Edge.of(array, o2, EdgeType.CURRENT_TRAVERSAL), "o2context");
    assertThat(receiver.objectContexts).containsEntry(o1, "o1context");
    assertThat(receiver.objectContexts).containsEntry(o2, "o2context");
  }

  @Test
  public void testObjectContexts() {
    Object o1 = new Object();
    Object o2 = new Object();
    Object pair = Pair.of(o1, o2);

    ConcurrentIdentitySet seen = new ConcurrentIdentitySet(1);
    LoggingObjectReceiver receiver = new LoggingObjectReceiver();
    DomainSpecificTraverser domainSpecific = mock(DomainSpecificTraverser.class);
    when(domainSpecific.admit(any())).thenReturn(true);
    when(domainSpecific.contextForField(refEq(pair), any(), any(), refEq(o1))).thenReturn("bad");
    when(domainSpecific.maybeTraverse(any(), any()))
        .thenAnswer(
            i -> {
              Object o = i.getArgument(0);
              Traversal traversal = i.getArgument(1);
              if (o == o1) {
                traversal.objectFound(o, "o1context");
                return true;
              } else if (o == o2) {
                traversal.objectFound(o, "o2context");
                return true;
              } else {
                return false;
              }
            });
    ObjectGraphTraverser cut = createObjectGraphTraverser(domainSpecific, seen, receiver, true);

    cut.traverse(pair);
    assertThat(receiver.objectContexts).containsEntry(o1, "o1context"); // overrides edge context
    assertThat(receiver.objectContexts).containsEntry(o2, "o2context");
  }
}
