// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.execgraph;

import static com.google.common.truth.Truth.assertThat;

import com.github.luben.zstd.ZstdOutputStream;
import com.google.devtools.build.execgraph.ExecGraphParser.FilteredStream;
import com.google.devtools.build.execgraph.ExecGraphParser.OrderedStream;
import com.google.devtools.build.lib.actions.ExecutionGraph.Node;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Testing ExecGraphParser */
@RunWith(JUnit4.class)
public final class ExecGraphParserTest {

  @Test
  public void readsZstdCompressedNodes() throws Exception {
    var path = Files.createTempFile("execgraph", ".tmp");
    try (var out = new ZstdOutputStream(Files.newOutputStream(path))) {
      Node.newBuilder().setDescription("action a").setMnemonic("Javac").build().writeDelimitedTo(out);
      Node.newBuilder().setDescription("action b").setMnemonic("CppCompile").build()
          .writeDelimitedTo(out);
    }

    try (var stream = ExecGraphParser.getMessageInputStream(path.toString())) {
      assertThat(stream.read())
          .isEqualTo(Node.newBuilder().setDescription("action a").setMnemonic("Javac").build());
      assertThat(stream.read())
          .isEqualTo(
              Node.newBuilder().setDescription("action b").setMnemonic("CppCompile").build());
      assertThat(stream.read()).isNull();
    }
  }

  @Test
  public void readEmpty() throws Exception {
    FilteredStream p = new FilteredStream(new FakeStream(new ArrayList<Node>()), null);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readEmptyWithRunner() throws Exception {
    FilteredStream p = new FilteredStream(new FakeStream(new ArrayList<Node>()), "local");
    assertThat(p.read()).isNull();
  }

  @Test
  public void readSingleNode() throws Exception {
    Node e = Node.newBuilder().setRunner("runs").setDescription("command").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e)), null);
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readSingleNodeRunnerMatch() throws Exception {
    Node e = Node.newBuilder().setRunner("runs").setDescription("command").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e)), "runs");
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readSingleNodeRunnerNoMatch() throws Exception {
    Node e = Node.newBuilder().setRunner("runs").setDescription("command").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e)), "run");
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyMatches() throws Exception {
    Node e = Node.newBuilder().setRunner("run1").setDescription("com1").build();
    Node e2 = Node.newBuilder().setRunner("r").setDescription("com2").build();
    Node e3 = Node.newBuilder().setRunner("run1").setDescription("com3").build();
    Node e4 = Node.newBuilder().setRunner("run1").setDescription("com4").build();
    Node e5 = Node.newBuilder().setRunner("ru").setDescription("com5").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3, e4, e5)), "run1");
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isEqualTo(e3);
    assertThat(p.read()).isEqualTo(e4);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyButNoMatch() throws Exception {
    Node e = Node.newBuilder().setRunner("run1").setDescription("com1").build();
    Node e2 = Node.newBuilder().setRunner("r").setDescription("com2").build();
    Node e3 = Node.newBuilder().setRunner("run1").setDescription("com3").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3)), "none");
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyNoMatcher() throws Exception {
    Node e = Node.newBuilder().setRunner("run1").setDescription("com1").build();
    Node e2 = Node.newBuilder().setRunner("r").setDescription("com2").build();
    Node e3 = Node.newBuilder().setRunner("run1").setDescription("com3").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3)), null);
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isEqualTo(e2);
    assertThat(p.read()).isEqualTo(e3);
    assertThat(p.read()).isNull();
  }

  private static class FakeStream implements MessageInputStream<Node> {
    List<Node> inputs;
    int i;

    public FakeStream(List<Node> ex) {
      this.inputs = ex;
      i = 0;
    }

    @Override
    public Node read() {
      if (i >= inputs.size()) {
        return null;
      }
      return inputs.get(i++);
    }

    @Override
    public void close() {}
  }

  public static FakeStream fakeParserFromStrings(List<String> strings) {
    ArrayList<Node> ins = new ArrayList<>(strings.size());
    for (String s : strings) {
      ins.add(Node.newBuilder().setDescription(s).setMnemonic(s).build());
    }
    return new FakeStream(ins);
  }

  public static OrderedStream.Golden getGolden(List<String> keys) {
    OrderedStream.Golden result = new OrderedStream.Golden();
    for (String s : keys) {
      result.addNode(Node.newBuilder().setDescription(s).build());
    }
    return result;
  }

  public void test(List<String> golden, List<String> input, List<String> expectedOutput)
      throws Exception {
    OrderedStream p = new OrderedStream(getGolden(golden), fakeParserFromStrings(input));

    List<String> got = new ArrayList<>();

    Node node;
    while ((node = p.read()) != null) {
      got.add(node.getDescription());
    }

    assertThat(got).containsExactlyElementsIn(expectedOutput).inOrder();
  }

  @Test
  public void reorderSimple() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("b", "a"), Arrays.asList("a", "b"));
  }

  @Test
  public void noReorderSimple() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("a", "b"), Arrays.asList("a", "b"));
  }

  @Test
  public void extraElement1() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("c", "b", "a"), Arrays.asList("a", "b", "c"));
  }

  @Test
  public void reorderExtraElement2() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("b", "c", "a"), Arrays.asList("a", "b", "c"));
  }

  @Test
  public void reorderMissingElement2() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("b"), Arrays.asList("b"));
  }

  @Test
  public void reorderLongMisc() throws Exception {
    test(
        Arrays.asList("a", "b", "c", "d"),
        Arrays.asList("y", "c", "b", "d", "x"),
        Arrays.asList("b", "c", "d", "y", "x"));
  }

  @Test
  public void reorderNothingThere() throws Exception {
    test(
        Arrays.asList("a", "b", "c", "d"),
        Arrays.asList("x", "y", "z"),
        Arrays.asList("x", "y", "z"));
  }

  @Test
  public void reorderPreservesInformation() throws Exception {
    List<String> golden = Arrays.asList("a", "b");

    // Include extra fields not present in golden.
    Node a = Node.newBuilder().setDescription("a").setMnemonic("acom").build();
    Node b = Node.newBuilder().setDescription("b").setMnemonic("bcom").build();
    Node c = Node.newBuilder().setDescription("c").setMnemonic("ccom").build();
    MessageInputStream<Node> input = new FakeStream(Arrays.asList(c, b, a));

    OrderedStream p = new OrderedStream(getGolden(golden), input);

    assertThat(p.read()).isEqualTo(a);
    assertThat(p.read()).isEqualTo(b);
    assertThat(p.read()).isEqualTo(c);
    assertThat(p.read()).isNull();
  }
}
