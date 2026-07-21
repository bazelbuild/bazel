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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.SettableWriteStatus;
import com.google.perftools.profiles.ProfileProto;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Location;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProfileRecorderTest {

  @Test
  public void buffering_mergesOnlyOnSuccessWithTrue() throws Exception {
    var collector = new ProfileCollector();
    var recorder = new ProfileRecorder(collector);

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);

    // Record some samples
    recorder.pushLocation(codecA());
    recorder.recordBytesAndPopLocation(0, codedOut); // count=1, bytes=0

    recorder.pushLocation(codecA());
    recorder.pushLocation(codecB());
    recorder.recordBytesAndPopLocation(0, codedOut); // count=1, bytes=0
    recorder.recordBytesAndPopLocation(
        0, codedOut); // count=1, bytes=0 (Wait, startBytes was 0, but codedOut moved)
    // Actually recordBytesAndPopLocation uses codedOut.getTotalBytesWritten() - startBytes.
    // If startBytes is 0 and we didn't write anything, byteCount is 0.

    // Collector should be empty
    assertThat(getSamples(collector.toProto())).isEmpty();

    // Trigger merge with false (not novel)
    recorder.onSuccess(false);
    assertThat(getSamples(collector.toProto())).isEmpty();

    // Create a new recorder for another batch
    recorder = new ProfileRecorder(collector);
    recorder.pushLocation(codecA());
    // Write 10 bytes
    codedOut.writeRawBytes(new byte[10]);
    recorder.recordBytesAndPopLocation(0, codedOut);

    // Trigger merge with true (novel)
    recorder.onSuccess(true);
    assertThat(getSamples(collector.toProto()))
        .containsExactly(new Sample(getStackText(codecA()), 1, 10));
  }

  @Test
  public void byteScale_scalesBytesOnMerge() throws Exception {
    var collector = new ProfileCollector();
    var recorder = new ProfileRecorder(collector);

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);

    recorder.pushLocation(codecA());
    codedOut.writeRawBytes(new byte[100]);
    recorder.recordBytesAndPopLocation(0, codedOut);

    recorder.setByteScale(0.5);
    recorder.onSuccess(true);

    assertThat(getSamples(collector.toProto()))
        .containsExactly(new Sample(getStackText(codecA()), 1, 50));
  }

  @Test
  public void registerWriteStatus_triggersMerge() throws Exception {
    var collector = new ProfileCollector();
    var recorder = new ProfileRecorder(collector);

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);

    recorder.pushLocation(codecA());
    codedOut.writeRawBytes(new byte[20]);
    recorder.recordBytesAndPopLocation(0, codedOut);

    var status = new SettableWriteStatus();
    recorder.registerWriteStatus(status);

    assertThat(getSamples(collector.toProto())).isEmpty();

    status.markSuccess(true);
    assertThat(getSamples(collector.toProto()))
        .containsExactly(new Sample(getStackText(codecA()), 1, 20));
  }

  @Test
  public void recordBytesAndPopLocation_worksWithoutCodedOutputStream() throws Exception {
    var collector = new ProfileCollector();
    var recorder = new ProfileRecorder(collector);

    recorder.pushLocation(codecA());
    recorder.recordBytes(50);
    recorder.popLocation();

    recorder.onSuccess(true);
    assertThat(getSamples(collector.toProto()))
        .containsExactly(new Sample(getStackText(codecA()), 1, 50));
  }

  private static CodecA codecA() {
    return CodecA.INSTANCE;
  }

  private static CodecB codecB() {
    return CodecB.INSTANCE;
  }

  private static ImmutableList<String> getStackText(ObjectCodec<?>... codecs) {
    var text = ImmutableList.<String>builder();
    for (var codec : codecs) {
      text.add(codec.getLocationText());
    }
    return text.build();
  }

  private record Sample(ImmutableList<String> stack, int count, int bytes) {}

  private static ImmutableList<Sample> getSamples(Profile profile) {
    List<String> strings = profile.getStringTableList();
    var functionNames = new HashMap<Integer, String>();
    for (var function : profile.getFunctionList()) {
      int id = (int) function.getId();
      functionNames.put(id, strings.get((int) function.getName()));
    }
    var locationNames = new HashMap<Integer, String>();
    for (Location location : profile.getLocationList()) {
      int id = (int) location.getId();
      List<Line> lines = location.getLineList();
      locationNames.put(id, functionNames.get((int) lines.get(0).getFunctionId()));
    }

    var samples = ImmutableList.<Sample>builder();
    for (ProfileProto.Sample sample : profile.getSampleList()) {
      var stack =
          sample.getLocationIdList().stream()
              .map(id -> locationNames.get((int) (long) id))
              .collect(toImmutableList());
      var values = sample.getValueList();
      samples.add(new Sample(stack, (int) (long) values.get(0), (int) (long) values.get(1)));
    }
    return samples.build();
  }

  private record A() {}

  private static class CodecA extends AsyncObjectCodec<A> {
    private static final CodecA INSTANCE = new CodecA();

    @Override
    public Class<A> getEncodedClass() {
      return A.class;
    }

    @Override
    public void serialize(SerializationContext c, A o, CodedOutputStream s) {}

    @Override
    public A deserializeAsync(AsyncDeserializationContext c, CodedInputStream s) {
      return null;
    }
  }

  private record B() {}

  private static class CodecB extends AsyncObjectCodec<B> {
    private static final CodecB INSTANCE = new CodecB();

    @Override
    public Class<B> getEncodedClass() {
      return B.class;
    }

    @Override
    public void serialize(SerializationContext c, B o, CodedOutputStream s) {}

    @Override
    public B deserializeAsync(AsyncDeserializationContext c, CodedInputStream s) {
      return null;
    }
  }
}
