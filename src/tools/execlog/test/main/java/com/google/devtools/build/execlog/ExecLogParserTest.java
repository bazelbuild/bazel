// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.execlog;

import static com.google.common.truth.Truth.assertThat;

import com.github.luben.zstd.ZstdOutputStream;
import com.google.devtools.build.execlog.ExecLogParser.FilteredStream;
import com.google.devtools.build.execlog.ExecLogParser.OrderedStream;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.exec.SpawnLogReconstructor;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.BinaryInputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.JsonInputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.BinaryOutputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.JsonOutputStreamWrapper;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Testing ExecLogParser */
@RunWith(JUnit4.class)
public final class ExecLogParserTest {

  @Test
  public void detectCompactFormat() throws Exception {
    var path = Files.createTempFile("compact", ".tmp");
    try (var out = new ZstdOutputStream(Files.newOutputStream(path))) {
      ExecLogEntry.newBuilder()
          .setInvocation(ExecLogEntry.Invocation.newBuilder().setHashFunctionName("SHA256"))
          .build()
          .writeDelimitedTo(out);
      ExecLogEntry.newBuilder()
          .setSpawn(ExecLogEntry.Spawn.newBuilder().addArgs("/bin/true"))
          .build()
          .writeDelimitedTo(out);
    }

    try (var stream = ExecLogParser.getMessageInputStream(path.toString())) {
      assertThat(stream).isInstanceOf(SpawnLogReconstructor.class);
      assertThat(stream.read())
          .isEqualTo(SpawnExec.newBuilder().addCommandArgs("/bin/true").build());
    }
  }

  @Test
  public void detectJsonFormat() throws Exception {
    var path = Files.createTempFile("json", ".tmp");
    try (var out = new JsonOutputStreamWrapper<SpawnExec>(Files.newOutputStream(path))) {
      out.write(SpawnExec.newBuilder().addCommandArgs("/bin/true").build());
    }

    try (var stream = ExecLogParser.getMessageInputStream(path.toString())) {
      assertThat(stream).isInstanceOf(JsonInputStreamWrapper.class);
      assertThat(stream.read())
          .isEqualTo(SpawnExec.newBuilder().addCommandArgs("/bin/true").build());
    }
  }

  @Test
  public void detectBinaryFormat() throws Exception {
    var path = Files.createTempFile("binary", ".tmp");
    try (var out = new BinaryOutputStreamWrapper<SpawnExec>(Files.newOutputStream(path))) {
      out.write(SpawnExec.newBuilder().addCommandArgs("/bin/true").build());
    }

    try (var stream = ExecLogParser.getMessageInputStream(path.toString())) {
      assertThat(stream).isInstanceOf(BinaryInputStreamWrapper.class);
      assertThat(stream.read())
          .isEqualTo(SpawnExec.newBuilder().addCommandArgs("/bin/true").build());
    }
  }

  @Test
  public void readEmpty() throws Exception {
    FilteredStream p = new FilteredStream(new FakeStream(new ArrayList<SpawnExec>()), null);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readEmptyWithRunner() throws Exception {
    FilteredStream p = new FilteredStream(new FakeStream(new ArrayList<SpawnExec>()), "local");
    assertThat(p.read()).isNull();
  }

  @Test
  public void readSingleSpawn() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e)), null);
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readSingleSpawnRunnerMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e)), "runs");
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readSingleSpawnRunnerNoMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e)), "run");
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyMatches() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3, e4, e5)), "run1");
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isEqualTo(e3);
    assertThat(p.read()).isEqualTo(e4);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyMatches1() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3, e4, e5)), "r");
    assertThat(p.read()).isEqualTo(e2);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyMatches2() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3, e4, e5)), "ru");
    assertThat(p.read()).isEqualTo(e5);
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyButNoMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3, e4, e5)), "none");
    assertThat(p.read()).isNull();
  }

  @Test
  public void readManyNoMatcher() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteredStream p = new FilteredStream(new FakeStream(Arrays.asList(e, e2, e3, e4, e5)), null);
    assertThat(p.read()).isEqualTo(e);
    assertThat(p.read()).isEqualTo(e2);
    assertThat(p.read()).isEqualTo(e3);
    assertThat(p.read()).isEqualTo(e4);
    assertThat(p.read()).isEqualTo(e5);
    assertThat(p.read()).isNull();
  }

  private static class FakeStream implements MessageInputStream<SpawnExec> {
    List<SpawnExec> inputs;
    int i;

    public FakeStream(List<SpawnExec> ex) {
      this.inputs = ex;
      i = 0;
    }

    @Override
    public SpawnExec read() {
      if (i >= inputs.size()) {
        return null;
      }
      return inputs.get(i++);
    }

    @Override
    public void close() {}
  };

  public static FakeStream fakeParserFromStrings(List<String> strings) {
    ArrayList<SpawnExec> ins = new ArrayList<>(strings.size());
    for (String s : strings) {
      ins.add(SpawnExec.newBuilder().addCommandArgs(s).addListedOutputs(s).build());
    }
    return new FakeStream(ins);
  }

  public static OrderedStream.Golden getGolden(List<String> keys) {
    OrderedStream.Golden result = new OrderedStream.Golden();
    for (String s : keys) {
      SpawnExec e = SpawnExec.newBuilder().addListedOutputs(s).build();
      result.addSpawnExec(e);
    }
    return result;
  }

  public void test(List<String> golden, List<String> input, List<String> expectedOutput)
      throws Exception {
    OrderedStream p = new OrderedStream(getGolden(golden), fakeParserFromStrings(input));

    List<String> got = new ArrayList<>();

    SpawnExec ex;
    while ((ex = p.read()) != null) {
      assertThat(ex.getCommandArgsCount()).isEqualTo(1);
      got.add(ex.getCommandArgs(0));
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
  public void reorderExtraElement3() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("b", "a", "c"), Arrays.asList("a", "b", "c"));
  }

  @Test
  public void reorderMissingElement1() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList(), Arrays.asList());
  }

  @Test
  public void reorderMissingElement2() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("b"), Arrays.asList("b"));
  }

  @Test
  public void reorderMissingElement3() throws Exception {
    test(Arrays.asList("a", "b"), Arrays.asList("a"), Arrays.asList("a"));
  }

  @Test
  public void reorderExtraElementOrder1() throws Exception {
    test(
        Arrays.asList("a", "b"),
        Arrays.asList("c", "a", "d", "b", "e"),
        Arrays.asList("a", "b", "c", "d", "e"));
  }

  @Test
  public void reorderExtraElementOrder2() throws Exception {
    test(
        Arrays.asList("a", "b"),
        Arrays.asList("c", "b", "d", "a", "e"),
        Arrays.asList("a", "b", "c", "d", "e"));
  }

  @Test
  public void reorderExtraElementOrder3() throws Exception {
    test(
        Arrays.asList("a", "b"),
        Arrays.asList("b", "c", "d", "e", "a"),
        Arrays.asList("a", "b", "c", "d", "e"));
  }

  @Test
  public void reorderExtraElementOrder4() throws Exception {
    test(
        Arrays.asList("a", "b"),
        Arrays.asList("b", "e", "d", "c", "a"),
        Arrays.asList("a", "b", "e", "d", "c"));
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
  public void reorderNothingThere2() throws Exception {
    test(Arrays.asList(), Arrays.asList("x", "y", "z"), Arrays.asList("x", "y", "z"));
  }

  @Test
  public void reorderPreservesInformation() throws Exception {
    List<String> golden = Arrays.asList("a", "b");

    // Include extra command arguments not present in golden
    SpawnExec a = SpawnExec.newBuilder().addListedOutputs("a").addCommandArgs("acom").build();
    SpawnExec b = SpawnExec.newBuilder().addListedOutputs("b").addCommandArgs("bcom").build();
    SpawnExec c = SpawnExec.newBuilder().addListedOutputs("c").addCommandArgs("ccom").build();
    MessageInputStream<SpawnExec> input = new FakeStream(Arrays.asList(c, b, a));

    OrderedStream p = new OrderedStream(getGolden(golden), input);

    assertThat(p.read()).isEqualTo(a);
    assertThat(p.read()).isEqualTo(b);
    assertThat(p.read()).isEqualTo(c);
    assertThat(p.read()).isNull();
  }
}
