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
package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec.stringCodec;
import static java.util.concurrent.ForkJoinPool.commonPool;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.strings.UnsafeStringCodec;
import com.google.errorprone.annotations.Keep;
import com.google.perftools.profiles.ProfileProto;
import com.google.perftools.profiles.ProfileProto.Line;
import com.google.perftools.profiles.ProfileProto.Location;
import com.google.perftools.profiles.ProfileProto.Profile;
import com.google.perftools.profiles.ProfileProto.ValueType;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ProfileCollectorTest {

  @Test
  public void toProto_hasExpectedMetadata() {
    var collector = new ProfileCollector();
    collector.recordSample(ImmutableList.of("a", "b"), 10);
    collector.recordSample(ImmutableList.of("a"), 20);

    Profile profile = collector.toProto();

    List<String> stringTable = profile.getStringTableList();
    assertThat(stringTable).hasSize(7);
    assertThat(stringTable.subList(0, 5))
        .isEqualTo(
            ImmutableList.of(
                "", // empty, required by the schema
                ProfileCollector.SAMPLES,
                ProfileCollector.COUNT,
                ProfileCollector.STORAGE,
                ProfileCollector.BYTES));
    // The records are traversed in a non-deterministic order. Depending on which one comes first,
    // "a" or "b" might be the earlier entry in the string table.
    assertThat(stringTable.subList(5, 7)).containsExactly("a", "b");

    assertThat(profile.getSampleTypeList())
        .containsExactly(
            // ProfileCollector.SAMPLES with units ProfileCollector.COUNT
            ValueType.newBuilder().setType(1).setUnit(2).build(),
            // ProfileCollector.STORAGE with units ProfileCollector.BYTES
            ValueType.newBuilder().setType(3).setUnit(4).build())
        .inOrder();

    assertThat(getSamples(profile))
        .containsExactly(
            // The stack trace is reversed with the leaf is position 0, as per the proto spec.
            new Sample(ImmutableList.of("b", "a"), 1, 10),
            // This was originally 20 but became 10 by subtracting the child.
            new Sample(ImmutableList.of("a"), 1, 10));
  }

  @Test
  public void toProto_aggregatesSamples() {
    var collector = new ProfileCollector();
    collector.recordSample(ImmutableList.of("a", "b", "c"), 10);
    collector.recordSample(ImmutableList.of("a", "b", "d"), 7);
    collector.recordSample(ImmutableList.of("a", "b"), 20);
    collector.recordSample(ImmutableList.of("a"), 25);

    collector.recordSample(ImmutableList.of("a", "b", "d"), 2);
    collector.recordSample(ImmutableList.of("a", "b"), 5);
    collector.recordSample(ImmutableList.of("a"), 10);

    collector.recordSample(ImmutableList.of("a"), 1);

    assertThat(getSamples(collector.toProto()))
        .containsExactly(
            // Only 1 entry. The stack trace is reversed with the leaf in position, as per the proto
            // spec.
            new Sample(ImmutableList.of("c", "b", "a"), 1, 10),
            // 2 samples, bytes = 2 + 7.
            new Sample(ImmutableList.of("d", "b", "a"), 2, 9),
            // 2 samples, bytes = 20 + 5 - (9 + 10) = 6.
            new Sample(ImmutableList.of("b", "a"), 2, 6),
            // 3 samples, bytes = 25 + 10 + 1 - (20 + 5) = 11.
            new Sample(ImmutableList.of("a"), 3, 11));
  }

  @Test
  public void memoizingCodec_profilingCorrectlyAccountsForBackreferences() throws Exception {
    // This test verifies that with the MemoizingSerializationContext, profiling correctly accounts
    // for memoized backreferences in the leaf and non-leaf case and nulls.

    // The example memoizes in a couple different places.
    var subject = new ArrayList<ExampleLeaf>();
    // 1. Initial item.
    subject.add(new ExampleLeaf("a", "b"));
    // 2. "a" is a memoized backreference to the 1st item's leaf "a".
    subject.add(new ExampleLeaf("a", "c"));
    // 3. Entire item will be a memoized backreference to the 1st item.
    subject.add(new ExampleLeaf("a", "b"));
    // 4. Exercises null leaves.
    subject.add(new ExampleLeaf(null, null));
    // 5. Exercises a null non-leaf.
    subject.add(null);

    var codecs = new ObjectCodecs();
    var profileCollector = new ProfileCollector();

    byte[] bytes =
        codecs.serializeMemoizedToBytes(
            subject, /* outputCapacity= */ 32, /* bufferSize= */ 32, profileCollector);
    assertThat(codecs.deserializeMemoized(bytes)).isEqualTo(subject); // sanity check

    ImmutableList<Sample> samples = getSamples(profileCollector.toProto());

    // Verifies the object counts of the samples. Exact byte counts are omitted to avoid
    // brittleness.
    ImmutableList<Sample> bytesErasedSamples =
        samples.stream()
            .map(sample -> new Sample(sample.stack(), sample.count(), 0))
            .collect(toImmutableList());
    assertThat(bytesErasedSamples)
        .containsExactly(
            new Sample(
                ImmutableList.of(ArrayListCodec.class.getCanonicalName()),
                1, // There's exactly 1 ArrayList.
                0),
            new Sample(
                ImmutableList.of(
                    ExampleLeafCodec.class.getCanonicalName(),
                    ArrayListCodec.class.getCanonicalName()),
                // The 4 samples here are the 1st-4th items. The null item doesn't increment the
                // count.
                4,
                0),
            new Sample(
                ImmutableList.of(
                    UnsafeStringCodec.class.getCanonicalName(),
                    ExampleLeafCodec.class.getCanonicalName(),
                    ArrayListCodec.class.getCanonicalName()),
                // The 6 samples here are 2 each from the 1st, 2nd and 4th list items. Memoized
                // leaves count as distinct samples. The 2 nulls in the 4th item can be counted as
                // two Strings because their type is known to the parent codec. The Strings in the
                // 3rd item are fully memoized away at the ExampleLeaf level.
                6,
                0));

    // Verifies that the profiler sees exactly the same number of bytes as output.
    int profiledBytes = samples.stream().mapToInt(Sample::bytes).sum();
    assertThat(profiledBytes).isEqualTo(bytes.length);
  }

  private record ExampleLeaf(String first, String second) {}

  @Keep
  private static class ExampleLeafCodec extends LeafObjectCodec<ExampleLeaf> {
    @Override
    public Class<ExampleLeaf> getEncodedClass() {
      return ExampleLeaf.class;
    }

    @Override
    public void serialize(
        LeafSerializationContext context, ExampleLeaf obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.first(), stringCodec(), codedOut);
      context.serializeLeaf(obj.second(), stringCodec(), codedOut);
    }

    @Override
    public ExampleLeaf deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      return new ExampleLeaf(
          context.deserializeLeaf(codedIn, stringCodec()),
          context.deserializeLeaf(codedIn, stringCodec()));
    }
  }

  @Test
  public void sharedValue_isOnlySerializedOnceAndInANewStack() throws Exception {
    var subject = new ExampleLeafSharer();
    subject.leaf = new ExampleLeaf("abc", "def");

    var codecs = new ObjectCodecs();
    var fingerprintValueService = FingerprintValueService.createForTesting();
    var profileCollector = new ProfileCollector();

    final int runCount = 20;

    AtomicInteger totalBytes = new AtomicInteger();
    var writeStatuses = Collections.synchronizedList(new ArrayList<ListenableFuture<Void>>());

    var allRunsDone = new CountDownLatch(runCount);
    for (int i = 0; i < runCount; i++) {
      commonPool()
          .execute(
              () -> {
                try {
                  SerializationResult<ByteString> result;
                  try {
                    result =
                        codecs.serializeMemoizedAndBlocking(
                            fingerprintValueService, subject, profileCollector);
                  } catch (SerializationException e) {
                    writeStatuses.add(immediateFailedFuture(e));
                    return;
                  }
                  totalBytes.getAndAdd(result.getObject().size());

                  ListenableFuture<Void> writeStatus = result.getFutureToBlockWritesOn();
                  if (writeStatus != null) {
                    writeStatuses.add(writeStatus);
                  }
                } finally {
                  allRunsDone.countDown();
                }
              });
    }
    allRunsDone.await();

    var unused = Futures.whenAllSucceed(writeStatuses).call(() -> null, directExecutor()).get();

    ImmutableList<Sample> samples = getSamples(profileCollector.toProto());

    var topStack = ImmutableList.<String>of(ExampleLeafSharerCodec.class.getCanonicalName());
    // Erases the bytes except for the top of the stack which is recorded in `totalBytes`. The other
    // bytes could be brittle to run assertions aren't easily recorded and would be brittle to
    // assert on.
    ImmutableList<Sample> bytesErasedSamples =
        samples.stream()
            .map(
                sample ->
                    sample.stack().equals(topStack)
                        ? sample
                        : new Sample(sample.stack(), sample.count(), 0))
            .collect(toImmutableList());
    assertThat(bytesErasedSamples)
        .containsExactly(
            //  The top level value is serialized runCount times and the bytes are precisely tracked
            //  in `totalBytes`.
            new Sample(topStack, runCount, totalBytes.get()),
            // The shared ExampleLeaf instance is only serialized once. Note that this is a shared
            // value, it is serialized under a new, independent stack.
            new Sample(ImmutableList.of(DeferredExampleLeafCodec.class.getCanonicalName()), 1, 0),
            new Sample(
                ImmutableList.of(
                    UnsafeStringCodec.class.getCanonicalName(),
                    DeferredExampleLeafCodec.class.getCanonicalName()),
                2, // "abc" and "def" in `subject.leaf`
                0));
  }

  private static class ExampleLeafSharer {
    private ExampleLeaf leaf; // mutable simplifies deserialization code

    private static void setLeaf(ExampleLeafSharer sharer, Object obj) {
      sharer.leaf = (ExampleLeaf) obj;
    }
  }

  @Keep
  private static class ExampleLeafSharerCodec extends AsyncObjectCodec<ExampleLeafSharer> {
    @Override
    public Class<ExampleLeafSharer> getEncodedClass() {
      return ExampleLeafSharer.class;
    }

    @Override
    public void serialize(
        SerializationContext context, ExampleLeafSharer obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.putSharedValue(
          obj.leaf, /* distinguisher= */ null, DeferredExampleLeafCodec.INSTANCE, codedOut);
    }

    @Override
    public ExampleLeafSharer deserializeAsync(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ExampleLeafSharer result = new ExampleLeafSharer();
      context.registerInitialValue(result);
      context.getSharedValue(
          codedIn,
          /* distinguisher= */ null,
          DeferredExampleLeafCodec.INSTANCE,
          result,
          ExampleLeafSharer::setLeaf);
      return result;
    }
  }

  /** As {@link DeferredObjectCodec} as required by {@link SerializationContext#putSharedValue}. */
  private static class DeferredExampleLeafCodec extends DeferredObjectCodec<ExampleLeaf> {
    private static final DeferredExampleLeafCodec INSTANCE = new DeferredExampleLeafCodec();

    @Override
    public Class<ExampleLeaf> getEncodedClass() {
      return ExampleLeaf.class;
    }

    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public void serialize(SerializationContext context, ExampleLeaf obj, CodedOutputStream codedOut)
        throws IOException, SerializationException {
      context.serializeLeaf(obj.first(), stringCodec(), codedOut);
      context.serializeLeaf(obj.second(), stringCodec(), codedOut);
    }

    @Override
    public DeferredValue<ExampleLeaf> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var result =
          new ExampleLeaf(
              context.deserializeLeaf(codedIn, stringCodec()),
              context.deserializeLeaf(codedIn, stringCodec()));
      return () -> result;
    }
  }

  private record Sample(ImmutableList<String> stack, int count, int bytes) {}

  /** Converts the {@code profile} message into an easily inspectable list of {@link Sample}s. */
  private static ImmutableList<Sample> getSamples(Profile profile) {
    List<String> strings = profile.getStringTableList();
    var functionNames = new HashMap<Integer, String>();
    for (var function : profile.getFunctionList()) {
      int id = (int) function.getId();
      String previous = functionNames.putIfAbsent(id, strings.get((int) function.getName()));
      assertWithMessage("duplicate function ID %s in %s", id, profile.getFunctionList())
          .that(previous)
          .isNull();
    }
    var locationNames = new HashMap<Integer, String>();
    for (Location location : profile.getLocationList()) {
      int id = (int) location.getId();
      List<Line> lines = location.getLineList();
      assertWithMessage("location with unexpected number of lines: %s", location)
          .that(lines)
          .hasSize(1);
      assertWithMessage("location with id different from function id: %s", location)
          .that(lines.get(0).getFunctionId())
          .isEqualTo(id);
      String previous = locationNames.putIfAbsent(id, functionNames.get(id));
      assertWithMessage("duplicate location ID %s in %s", id, profile.getLocationList())
          .that(previous)
          .isNull();
    }
    assertThat(locationNames).isEqualTo(functionNames);

    var samples = ImmutableList.<Sample>builder();
    for (ProfileProto.Sample sample : profile.getSampleList()) {
      var stack =
          sample.getLocationIdList().stream()
              .map(id -> locationNames.get((int) (long) id))
              .collect(toImmutableList());
      var values = sample.getValueList();
      assertThat(values).hasSize(2);
      samples.add(new Sample(stack, (int) (long) values.get(0), (int) (long) values.get(1)));
    }
    return samples.build();
  }
}
