// Copyright 2019 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.bazel.execlog;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.protobuf.Message;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StableSort}. */
@RunWith(JUnit4.class)
public final class StableSortTest {

  private static class ListOutput implements MessageOutputStream {
    public ArrayList<SpawnExec> list;

    ListOutput() {
      list = new ArrayList<>();
    }

    @Override
    public void write(Message m) throws IOException {
      Preconditions.checkNotNull(m);
      Preconditions.checkArgument(m instanceof SpawnExec);
      list.add((SpawnExec) m);
    }

    @Override
    public void close() throws IOException {}
  }

  ArrayList<SpawnExec> testStableSort(List<SpawnExec> list) throws Exception {
    ListOutput o = new ListOutput();
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    for (SpawnExec spawn : list) {
      spawn.writeDelimitedTo(baos);
    }
    InputStream inputStream = new ByteArrayInputStream(baos.toByteArray());

    StableSort.stableSort(inputStream, o);
    return o.list;
  }

  private static SpawnExec.Builder createSpawnExecBuilder(
      List<String> inputs, List<String> outputs) {
    SpawnExec.Builder e = SpawnExec.newBuilder();
    for (String output : outputs) {
      e.addActualOutputsBuilder().setPath(output);
      e.addListedOutputs(output);
    }
    for (String s : inputs) {
      e.addInputs(File.newBuilder().setPath(s).build());
    }
    return e;
  }

  private static SpawnExec createSpawnExec(List<String> inputs, List<String> outputs) {
    return createSpawnExecBuilder(inputs, outputs).build();
  }

  @Test
  public void stableSortEmpty() throws Exception {
    List<SpawnExec> l = testStableSort(ImmutableList.of());
    assertThat(l).isEmpty();
  }

  @Test
  public void stableSortOne() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of(), ImmutableList.of("output"));
    List<SpawnExec> l = testStableSort(ImmutableList.of(e1));
    assertThat(l).containsExactly(e1).inOrder();
  }

  @Test
  public void stableSortTwo_unlinkedLexicographic() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("a"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("leaf2"), ImmutableList.of("b"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e1, e2));
    assertThat(l).containsExactly(e1, e2).inOrder();
  }

  @Test
  public void stableSortTwo_unlinkedLexicographic_reverse() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("b"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("leaf2"), ImmutableList.of("a"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e1, e2));
    assertThat(l).containsExactly(e2, e1).inOrder();
  }

  @Test
  public void stableSortTwo_linked() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("b"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("b"), ImmutableList.of("a"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e1, e2));
    assertThat(l).containsExactly(e1, e2).inOrder();
  }

  @Test
  public void stableSortTwo_linked_inputOrderDoesNotMatter() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("b"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("b"), ImmutableList.of("a"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e2, e1));
    assertThat(l).containsExactly(e1, e2).inOrder();
  }

  @Test
  public void stableSortTwo_oneOfMultipleInputs() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("b1", "b2", "b3"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("b2"), ImmutableList.of("a"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e2, e1));
    assertThat(l).containsExactly(e1, e2).inOrder();
  }

  @Test
  public void stableSortTwo_manyOfMultipleInputs() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("b1", "b2", "b3"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("b2", "b3"), ImmutableList.of("a"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e2, e1));
    assertThat(l).containsExactly(e1, e2).inOrder();
  }

  @Test
  public void stableSortTwo_IrrelevantInputs() throws Exception {
    SpawnExec e1 = createSpawnExec(ImmutableList.of("leaf1"), ImmutableList.of("b1", "b2", "b3"));
    SpawnExec e2 = createSpawnExec(ImmutableList.of("z", "b2", "1"), ImmutableList.of("a"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(e2, e1));
    assertThat(l).containsExactly(e1, e2).inOrder();
  }

  @Test
  public void stableSortTwo_ABC() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of(""), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of(""), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c));
    assertThat(l).containsExactly(a, b, c).inOrder();
  }

  @Test
  public void stableSortTwo_CBA() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("b"), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of("c"), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c));
    assertThat(l).containsExactly(c, b, a).inOrder();
  }

  @Test
  public void stableSortTwo_ACB() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of(""), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of("a", "c"), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c));
    assertThat(l).containsExactly(a, c, b).inOrder();
  }

  @Test
  public void stableSortTwo_CAB() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("c"), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of("c"), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c));
    assertThat(l).containsExactly(c, a, b).inOrder();
  }

  @Test
  public void stableSortTwo_CAB2() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("c1"), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of("c2"), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c1", "c2"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c));
    assertThat(l).containsExactly(c, a, b).inOrder();
  }

  @Test
  public void stableSortTwo_CBAFED() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("b"), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of("c"), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c"));
    SpawnExec d = createSpawnExec(ImmutableList.of("e"), ImmutableList.of("d"));
    SpawnExec e = createSpawnExec(ImmutableList.of("f"), ImmutableList.of("e"));
    SpawnExec f = createSpawnExec(ImmutableList.of(""), ImmutableList.of("f"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c, d, e, f));
    assertThat(l).containsExactly(c, b, a, f, e, d).inOrder();
  }

  @Test
  public void stableSortTwo_InterleavedPaths() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("c"), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of(""), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of(""), ImmutableList.of("c"));
    SpawnExec d = createSpawnExec(ImmutableList.of("a"), ImmutableList.of("d"));
    SpawnExec e = createSpawnExec(ImmutableList.of("f"), ImmutableList.of("e"));
    SpawnExec f = createSpawnExec(ImmutableList.of("b"), ImmutableList.of("f"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c, d, e, f));
    assertThat(l).containsExactly(b, c, a, d, f, e).inOrder();
  }

  @Test
  public void stableSortTwo_ManyDependencies() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("b", "c", "f"), ImmutableList.of("a"));
    SpawnExec b = createSpawnExec(ImmutableList.of("d", "e"), ImmutableList.of("b"));
    SpawnExec c = createSpawnExec(ImmutableList.of("e", "d", "f"), ImmutableList.of("c"));
    SpawnExec d = createSpawnExec(ImmutableList.of(""), ImmutableList.of("d"));
    SpawnExec e = createSpawnExec(ImmutableList.of("f"), ImmutableList.of("e"));
    SpawnExec f = createSpawnExec(ImmutableList.of(""), ImmutableList.of("f"));

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c, d, e, f));
    assertThat(l).containsExactly(d, f, e, b, c, a).inOrder();
  }

  @Test
  public void stableSort_NoOutputs() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("a"), ImmutableList.of());
    SpawnExec b = createSpawnExec(ImmutableList.of("b"), ImmutableList.of());

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b));
    assertThat(l).containsExactly(a, b).inOrder();
  }

  @Test
  public void stableSort_NoOutputs_reversed() throws Exception {
    SpawnExec a = createSpawnExec(ImmutableList.of("a"), ImmutableList.of());
    SpawnExec b = createSpawnExec(ImmutableList.of("b"), ImmutableList.of());

    List<SpawnExec> l = testStableSort(ImmutableList.of(b, a));
    assertThat(l).containsExactly(a, b).inOrder();
  }

  @Test
  public void stableSort_ListedOutputs() throws Exception {

    SpawnExec a =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("a"))
            .addCommandArgs("a")
            .build();
    SpawnExec b =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("b").build();
    SpawnExec c =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("c"))
            .addCommandArgs("c")
            .build();
    SpawnExec d =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("d"))
            .addCommandArgs("d")
            .build();
    SpawnExec e =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("e").build();
    SpawnExec f =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("f").build();

    List<SpawnExec> l = testStableSort(ImmutableList.of(a, b, c, d, e, f));
    assertThat(l)
        .containsExactly(
            // sorted elements with actual outputs
            a,
            c,
            d,
            // sorted elements without listed outputs
            b,
            e,
            f)
        .inOrder();
  }

  @Test
  public void stableSort_ListedOutputs_reordered() throws Exception {
    SpawnExec a =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("a"))
            .addCommandArgs("a")
            .build();
    SpawnExec b =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("b").build();
    SpawnExec c =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("c"))
            .addCommandArgs("c")
            .build();
    SpawnExec d =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("d"))
            .addCommandArgs("d")
            .build();
    SpawnExec e =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("e").build();
    SpawnExec f =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("f").build();

    // Reordering the input from the previous test does not change the resulting order
    List<SpawnExec> l = testStableSort(ImmutableList.of(f, e, d, c, b, a));
    assertThat(l)
        .containsExactly(
            // sorted elements with actual outputs
            a,
            c,
            d,
            // sorted elements without listed outputs
            b,
            e,
            f)
        .inOrder();
  }

  @Test
  public void stableSort_ListedOutputs_dependencies() throws Exception {
    // Dependencies are respected
    SpawnExec a =
        createSpawnExecBuilder(ImmutableList.of("d"), ImmutableList.of("a"))
            .addCommandArgs("a")
            .build();
    SpawnExec b =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("b").build();
    SpawnExec c =
        createSpawnExecBuilder(ImmutableList.of("d"), ImmutableList.of("c"))
            .addCommandArgs("c")
            .build();
    SpawnExec d =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of("d"))
            .addCommandArgs("d")
            .build();
    SpawnExec e =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("e").build();
    SpawnExec f =
        createSpawnExecBuilder(ImmutableList.of(), ImmutableList.of()).addCommandArgs("f").build();

    List<SpawnExec> l = testStableSort(ImmutableList.of(f, e, d, c, b, a));
    assertThat(l).containsExactly(d, a, c, b, e, f).inOrder();
  }
}
