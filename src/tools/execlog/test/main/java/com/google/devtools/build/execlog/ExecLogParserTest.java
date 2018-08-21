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

import com.google.devtools.build.execlog.ExecLogParser.FilteringLogParser;
import com.google.devtools.build.execlog.ExecLogParser.Parser;
import com.google.devtools.build.execlog.ExecLogParser.ReorderingParser;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Testing ExecLogParser */
@RunWith(JUnit4.class)
public final class ExecLogParserTest {

  private InputStream toInputStream(List<SpawnExec> list) throws Exception {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    for (SpawnExec spawnExec : list) {
      spawnExec.writeDelimitedTo(bos);
    }

    return new ByteArrayInputStream(bos.toByteArray());
  }

  @Test
  public void getNextEmpty() throws Exception {
    FilteringLogParser p = new FilteringLogParser(toInputStream(new ArrayList<SpawnExec>()), null);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextEmptyWithRunner() throws Exception {
    FilteringLogParser p =
        new FilteringLogParser(toInputStream(new ArrayList<SpawnExec>()), "local");
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleSpawn() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    FilteringLogParser p = new FilteringLogParser(toInputStream(Arrays.asList(e)), null);
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleSpawnRunnerMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    FilteringLogParser p = new FilteringLogParser(toInputStream(Arrays.asList(e)), "runs");
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleSpawnRunnerNoMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    FilteringLogParser p = new FilteringLogParser(toInputStream(Arrays.asList(e)), "run");
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyMatches() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteringLogParser p =
        new FilteringLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "run1");
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isEqualTo(e3);
    assertThat(p.getNext()).isEqualTo(e4);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyMatches1() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteringLogParser p =
        new FilteringLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "r");
    assertThat(p.getNext()).isEqualTo(e2);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyMatches2() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteringLogParser p =
        new FilteringLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "ru");
    assertThat(p.getNext()).isEqualTo(e5);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyButNoMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteringLogParser p =
        new FilteringLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "none");
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyNoMatcher() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    FilteringLogParser p =
        new FilteringLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), null);
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isEqualTo(e2);
    assertThat(p.getNext()).isEqualTo(e3);
    assertThat(p.getNext()).isEqualTo(e4);
    assertThat(p.getNext()).isEqualTo(e5);
    assertThat(p.getNext()).isNull();
  }

  private static class FakeParser implements Parser {
    List<SpawnExec> inputs;
    int i;

    public FakeParser(List<SpawnExec> ex) {
      this.inputs = ex;
      i = 0;
    }

    @Override
    public SpawnExec getNext() {
      if (i >= inputs.size()) {
        return null;
      }
      return inputs.get(i++);
    }
  };

  public static FakeParser fakeParserFromStrings(List<String> strings) {
    ArrayList<SpawnExec> ins = new ArrayList<>(strings.size());
    for (String s : strings) {
      ins.add(SpawnExec.newBuilder().addCommandArgs(s).addListedOutputs(s).build());
    }
    return new FakeParser(ins);
  }

  public static ReorderingParser.Golden getGolden(List<String> keys) {
    ReorderingParser.Golden result = new ReorderingParser.Golden();
    for (String s : keys) {
      SpawnExec e = SpawnExec.newBuilder().addListedOutputs(s).build();
      result.addSpawnExec(e);
    }
    return result;
  }

  public void test(List<String> golden, List<String> input, List<String> expectedOutput)
      throws Exception {
    ReorderingParser p = new ReorderingParser(getGolden(golden), fakeParserFromStrings(input));

    List<String> got = new ArrayList<>();

    SpawnExec ex;
    while ((ex = p.getNext()) != null) {
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
    Parser input = new FakeParser(Arrays.asList(c, b, a));

    ReorderingParser p = new ReorderingParser(getGolden(golden), input);

    assertThat(p.getNext()).isEqualTo(a);
    assertThat(p.getNext()).isEqualTo(b);
    assertThat(p.getNext()).isEqualTo(c);
    assertThat(p.getNext()).isNull();
  }
}

