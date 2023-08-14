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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.skyframe.InMemoryGraphImpl.EdgelessInMemoryGraphImpl;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import org.junit.Test;

/** Tests for {@link InMemoryGraphImpl}. */
public class InMemoryGraphTest extends GraphTest {

  @Override
  protected Version getStartingVersion() {
    return IntVersion.of(0);
  }

  @Override
  protected Version getNextVersion(Version v) {
    Preconditions.checkState(v instanceof IntVersion);
    return ((IntVersion) v).next();
  }

  @Override
  protected void makeGraph() {
    graph = new InMemoryGraphImpl();
  }

  @Override
  protected ProcessableGraph getGraph(Version version) {
    return graph;
  }

  /** Tests for {@link EdgelessInMemoryGraphImpl}. */
  public static final class EdgelessInMemoryGraphTest extends InMemoryGraphTest {

    @Override
    protected void makeGraph() {}

    @Override
    protected ProcessableGraph getGraph(Version version) {
      return new EdgelessInMemoryGraphImpl(/* usePooledInterning= */ true);
    }

    @Override
    protected Version getStartingVersion() {
      return Version.constant();
    }

    @Override
    protected Version getNextVersion(Version version) {
      throw new UnsupportedOperationException();
    }

    @Override
    protected boolean shouldTestIncrementality() {
      return false;
    }
  }

  public static final class SkyKeyWithSkyKeyInterner extends AbstractSkyKey<String> {
    private static final SkyKeyInterner<SkyKeyWithSkyKeyInterner> interner = SkyKey.newInterner();

    static SkyKeyWithSkyKeyInterner create(String arg) {
      return interner.intern(new SkyKeyWithSkyKeyInterner(arg));
    }

    private SkyKeyWithSkyKeyInterner(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }

    @Override
    public SkyKeyInterner<SkyKeyWithSkyKeyInterner> getSkyKeyInterner() {
      return interner;
    }
  }

  @Test
  public void createIfAbsentBatch_skyKeyWithSkyKeyInterner() throws InterruptedException {
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");

    // Insert cat SkyKey into graph.
    // (1) result of getting cat node from graph should not be null;
    // (2) when re-creating cat SkyKeyWithSkyKeyInterner object, it should retrieve the instance
    // from global pool (graph), which is also the same instance as the original one.
    graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(cat));
    assertThat(graph.get(null, Reason.OTHER, cat)).isNotNull();
    assertThat(SkyKeyWithSkyKeyInterner.create("cat")).isSameInstanceAs(cat);

    // Remove cat SkyKey from graph.
    // (1) result of getting cat node from graph should be null, indicating the cat key has been
    // removed from the global pool (graph);
    // (2) since when removing key from global pool (graph), the removed key will be re-interned
    // back to weak interner. So re-creating an equal "cat" object from SkyKeyWithSkyKeyInterner
    // will result in the same instance to be returned (no new instance will be created).
    graph.remove(cat);
    assertThat(graph.get(null, Reason.OTHER, cat)).isNull();
    assertThat(SkyKeyWithSkyKeyInterner.create("cat")).isSameInstanceAs(cat);
  }

  @Test
  public void cleanupPool_weakInternerReintern() throws InterruptedException {
    SkyKey cat = SkyKeyWithSkyKeyInterner.create("cat");

    graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(cat));
    assertThat(graph.get(null, Reason.OTHER, cat)).isNotNull();

    assertThat(graph).isInstanceOf(InMemoryGraphImpl.class);
    ((InMemoryGraphImpl) graph).cleanupInterningPools();

    // When re-creating a cat SkyKeyWithSkyKeyInterner, we expect to get the original instance. Pool
    // cleaning up re-interns the cat instance back to the weak interner, and thus, no new instance
    // is created.
    assertThat(SkyKeyWithSkyKeyInterner.create("cat")).isSameInstanceAs(cat);
  }

  @Test
  public void removePackageNode_notPresentInGraph() throws Exception {
    PackageIdentifier packageIdentifier = PackageIdentifier.createUnchecked("repo", "hello");

    graph.remove(packageIdentifier);
    assertThat(graph.get(null, Reason.OTHER, packageIdentifier)).isNull();
  }

  @Test
  public void removePackageNode_noValueWeakInternLabelsNoCrash() throws Exception {
    PackageIdentifier packageIdentifier = PackageIdentifier.createUnchecked("repo", "hello");

    graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(packageIdentifier));
    NodeEntry entry = graph.get(null, Reason.OTHER, packageIdentifier);
    assertThat(entry.toValue()).isNull();

    graph.remove(packageIdentifier);
    assertThat(graph.get(null, Reason.OTHER, packageIdentifier)).isNull();
  }
}
