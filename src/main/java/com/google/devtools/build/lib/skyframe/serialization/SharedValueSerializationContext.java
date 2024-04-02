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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.aggregateStatusFutures;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.getDoneSerializationFuture;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.reportAnyFailures;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.waitForSerializationFuture;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A {@link SerializationContext} that supports both memoization and shared subobjects.
 *
 * <p>Sharing subobjects occurs by means of uploading them asynchronously to a {@link
 * #fingerprintValueService} store. The status of these uploads may be observed through {@link
 * #createFutureToBlockWritingOn} and {@link SerializationResult#getFutureToBlockWritesOn}.
 */
final class SharedValueSerializationContext extends MemoizingSerializationContext {
  private final FingerprintValueService fingerprintValueService;

  /**
   * Futures that represent writes to remote storage.
   *
   * <p>For consistency, the serialized bytes should not be published for other consumers until
   * these writes complete.
   */
  @Nullable // lazily initialized
  private ArrayList<ListenableFuture<Void>> futuresToBlockWritingOn;

  /**
   * Futures that mark deferred data bytes when using {@link #putSharedValue}.
   *
   * <p>The serialized byte representation contains placeholders until these futures complete.
   */
  @Nullable // lazily initialized
  private ArrayList<FuturePut> futurePuts;

  @VisibleForTesting // private
  static SharedValueSerializationContext createForTesting(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService) {
    return new SharedValueSerializationContext(
        codecRegistry, dependencies, fingerprintValueService);
  }

  private SharedValueSerializationContext(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService) {
    super(codecRegistry, dependencies);
    this.fingerprintValueService = fingerprintValueService;
  }

  /**
   * Serializes {@code subject} and returns a result that may have an associated future.
   *
   * <p>This method may block when the serialization of a shared child is performed by another
   * thread. It blocks until the fingerprints of such shared children become available. However,
   * this method does not block on uploads to the {@link FingerprintValueService}. The status of the
   * upload is provided by {@link SerializationResult#getFutureToBlockWritesOn}.
   */
  static SerializationResult<ByteString> serializeToResult(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService,
      @Nullable Object subject)
      throws SerializationException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    SharedValueSerializationContext context =
        new SharedValueSerializationContext(codecRegistry, dependencies, fingerprintValueService);
    try {
      context.serialize(subject, codedOut);
      codedOut.flush();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize: " + subject, e);
    }
    return context.createResult(bytesOut.toByteArray());
  }

  @Override
  public <T> void putSharedValue(
      T child,
      @Nullable Object distinguisher,
      DeferredObjectCodec<T> codec,
      CodedOutputStream codedOut)
      throws IOException, SerializationException {
    SettableFuture<PutOperation> putOperation = SettableFuture.create();
    Object previous =
        fingerprintValueService.getOrClaimPutOperation(child, distinguisher, putOperation);
    if (previous != null) {
      // Uses the previous result and discards `putOperation`.
      if (previous instanceof ListenableFuture) {
        // HINT: this is the only place where `FuturePut`s originate. The other caller of
        // `recordFuturePut` propagates them transitively.
        @SuppressWarnings("unchecked")
        ListenableFuture<PutOperation> inflight = (ListenableFuture<PutOperation>) previous;
        recordFuturePut(inflight, codedOut);
        return;
      }
      codedOut.writeRawBytes((ByteString) previous);
      return;
    }

    try {
      putOwnedSharedValue(child, codec, codedOut, putOperation);
    } catch (SerializationException | IOException e) {
      putOperation.setException(e);
      throw e;
    } catch (RuntimeException | Error e) {
      // `putOperation` must always be set to avoid deadlock-like behaviors in its consumers.
      putOperation.setException(e);
      throw e;
    }
  }

  /**
   * Implementation of {@link #putSharedValue} when it owns the {@code putOperation}.
   *
   * <p>This method does the following.
   *
   * <ul>
   *   <li>Uploads {@code child}'s bytes into {@link #fingerprintValueService}.
   *   <li>Sets the result of the upload in {@code putOperation}.
   *   <li>Writes the fingerprint of {@code child}'s serialized bytes (or a placeholder) to {@code
   *       codedOut}.
   *   <li>Adds either a {@link #futuresToBlockWritingOn} entry on {@link #futurePuts} entry.
   * </ul>
   *
   * <p>There are two cases, in particular.
   *
   * <ul>
   *   <li><em>Serializing {@code child} causes recursive calls to {@link #putSharedValue} and some
   *       of those recursive calls are are still being processed in a different thread</em>: this
   *       results in placeholders being added to the {@code codedOut} and {@link #futurePuts}
   *       entries to be recorded.
   *   <li><em>Any other case, for example, serializing {@code child} does not recursively call
   *       {@link #putSharedValue} or if it does, the fingerprints are either already available or
   *       owned by this thread</em>: here, the serialized bytes of {@code child} are computed in
   *       the calling thread and become immediately available. The status of uploading the
   *       serialized {@code child} is added to {@link #futuresToBlockWritingOn}.
   * </ul>
   *
   * <p>When entries are added to {@link #futurePuts}, the caller must ensure that for every entry,
   * {@link FuturePut#fillInPlaceholderAndReturnWriteStatus} is called to update placeholders
   * present in the serialized bytes.
   */
  private <T> void putOwnedSharedValue(
      T child,
      DeferredObjectCodec<T> codec,
      CodedOutputStream codedOut,
      SettableFuture<PutOperation> putOperation)
      throws IOException, SerializationException {
    byte[] childBytes; // bytes of serialized child, maybe containing placeholders
    // This is initially populated with the `futuresToWriteBlockingOn` of the child. If the child
    // has `FuturePut`s, the write statuses of those puts will be added to this list.
    ArrayList<ListenableFuture<Void>> childWriteStatuses;
    // Each entry of the following list corresponds to a `putSharedValue` call made by serialization
    // of `child`. Used to fill-in `childBytes` placeholders.
    ArrayList<FuturePut> childDeferredBytes;
    { // Serializes `child` with a fresh context, populating the above variables.
      ByteArrayOutputStream childStream = new ByteArrayOutputStream();
      CodedOutputStream childCodedOut = CodedOutputStream.newInstance(childStream);
      SharedValueSerializationContext childContext = getFreshContext();
      codec.serialize(childContext, child, childCodedOut);
      childCodedOut.flush();

      childBytes = childStream.toByteArray();
      childWriteStatuses =
          childContext.futuresToBlockWritingOn == null
              ? new ArrayList<>()
              : childContext.futuresToBlockWritingOn;
      childDeferredBytes = childContext.futurePuts;
    }

    if (childDeferredBytes == null) {
      // There are no deferred bytes so `childBytes` is complete. Starts the upload.
      ByteString fingerprint = fingerprintValueService.fingerprint(childBytes);
      codedOut.writeRawBytes(fingerprint); // Writes only the fingerprint to the stream.
      childWriteStatuses.add(fingerprintValueService.put(fingerprint, childBytes));

      ListenableFuture<Void> writeStatus = aggregateStatusFutures(childWriteStatuses);
      putOperation.set(PutOperation.create(fingerprint, writeStatus));
      addFutureToBlockWritingOn(writeStatus);
      return;
    }

    // Ensures enough space for the subvalues and one additional for the put of `child` itself.
    childWriteStatuses.ensureCapacity(childWriteStatuses.size() + childDeferredBytes.size() + 1);

    // There are pending fingerprint sub-computations.
    ListenableFuture<PutOperation> futureStore =
        whenAllPutsStarted(childDeferredBytes)
            .call(
                () -> {
                  fillPlaceholdersAndCollectWriteStatuses(
                      childDeferredBytes, childBytes, childWriteStatuses);
                  // All placeholders are filled-in. Starts the upload.
                  ByteString fingerprint = fingerprintValueService.fingerprint(childBytes);
                  childWriteStatuses.add(fingerprintValueService.put(fingerprint, childBytes));
                  return PutOperation.create(
                      fingerprint, aggregateStatusFutures(childWriteStatuses));
                },
                directExecutor());

    putOperation.setFuture(futureStore);
    recordFuturePut(futureStore, codedOut);
  }

  @Override
  public SharedValueSerializationContext getFreshContext() {
    return new SharedValueSerializationContext(
        getCodecRegistry(), getDependencies(), fingerprintValueService);
  }

  @Override
  public void addFutureToBlockWritingOn(ListenableFuture<Void> future) {
    if (futuresToBlockWritingOn == null) {
      futuresToBlockWritingOn = new ArrayList<>();
    }
    futuresToBlockWritingOn.add(future);
  }

  @Override
  @Nullable
  public ListenableFuture<Void> createFutureToBlockWritingOn() {
    if (futurePuts == null) {
      if (futuresToBlockWritingOn == null) {
        return null;
      }
      return aggregateStatusFutures(futuresToBlockWritingOn);
    }

    List<ListenableFuture<Void>> writeStatuses = initializeCombinedWriteStatusesList();
    for (FuturePut futurePut : futurePuts) {
      writeStatuses.add(
          Futures.transformAsync(
              futurePut.getFuturePut(), PutOperation::writeStatus, directExecutor()));
    }
    return aggregateStatusFutures(writeStatuses);
  }

  /**
   * Records a deferred fingerprint value to be written to the serialized bytes.
   *
   * <p>Adds the corresponding entry to {@link #futurePuts} and inserts placeholder bytes into
   * {@code codedOut}.
   */
  private void recordFuturePut(ListenableFuture<PutOperation> futurePut, CodedOutputStream codedOut)
      throws IOException {
    if (futurePuts == null) {
      futurePuts = new ArrayList<>();
    }
    futurePuts.add(new FuturePut(codedOut.getTotalBytesWritten(), futurePut));
    // Adds a placeholder for the real fingerprint to be filled in after the future completes.
    codedOut.writeRawBytes(fingerprintValueService.fingerprintPlaceholder());
  }

  private SerializationResult<ByteString> createResult(byte[] bytes) throws SerializationException {
    // TODO: b/297857068 - If ByteString.copyFrom overhead is excessive, use reflection to avoid it.
    if (futurePuts == null) {
      ByteString finalBytes = ByteString.copyFrom(bytes);
      return futuresToBlockWritingOn == null
          ? SerializationResult.createWithoutFuture(finalBytes)
          : SerializationResult.create(finalBytes, aggregateStatusFutures(futuresToBlockWritingOn));
    }

    List<ListenableFuture<Void>> writeStatuses = initializeCombinedWriteStatusesList();
    // `futurePuts` are produced when multiple threads race to write the same value. This is
    // expected to be relatively rare and waiting here is for CPU bound serialization work only.
    //
    // TODO: b/297857068 - It's possible to expose this intermediate future to callers as a
    // SerializationResult<ListenableFuture<byte[]>>. Do so if the contention outweighs the extra
    // complexity.
    try {
      waitForSerializationFuture(
          whenAllPutsStarted(futurePuts)
              .call(
                  () -> {
                    fillPlaceholdersAndCollectWriteStatuses(futurePuts, bytes, writeStatuses);
                    return null;
                  },
                  directExecutor()));
    } catch (SerializationException e) {
      // An exception has occurred. Ensures that any additional errors from writing are handled
      // before throwing the exception.
      reportAnyFailures(Futures.whenAllSucceed(futuresToBlockWritingOn));
      throw e;
    }
    return SerializationResult.create(
        ByteString.copyFrom(bytes), aggregateStatusFutures(writeStatuses));
  }

  /**
   * Initializes a write status list including any {@link #futuresToBlockWritingOn} and sized for
   * the caller to add statuses from {@link #futurePuts}.
   *
   * <p>The caller must ensure that {@link #futurePuts} is non-null.
   */
  private List<ListenableFuture<Void>> initializeCombinedWriteStatusesList() {
    if (futuresToBlockWritingOn == null) {
      return new ArrayList<>(futurePuts.size());
    }
    ArrayList<ListenableFuture<Void>> statuses =
        new ArrayList<>(futurePuts.size() + futuresToBlockWritingOn.size());
    statuses.addAll(futuresToBlockWritingOn);
    return statuses;
  }

  private static Futures.FutureCombiner<?> whenAllPutsStarted(List<FuturePut> stores) {
    return Futures.whenAllSucceed(Lists.transform(stores, FuturePut::getFuturePut));
  }

  /**
   * Fills placeholders in {@code bytes} from {@code futurePuts} and adds write statuses to {@code
   * writeStatuses}.
   *
   * <p>The caller must ensure that all the {@link FuturePut#futurePut}s futures in {@code
   * futurePuts} are done before calling this, that is, {@link #whenAllPutsStarted}.
   */
  private static void fillPlaceholdersAndCollectWriteStatuses(
      List<FuturePut> futurePuts, byte[] bytes, List<ListenableFuture<Void>> writeStatuses)
      throws SerializationException {
    for (FuturePut put : futurePuts) {
      writeStatuses.add(put.fillInPlaceholderAndReturnWriteStatus(bytes));
    }
  }

  /**
   * A tuple consisting of a stream offset and a pending {@link PutOperation}.
   *
   * <p>When {@link PutOperation} becomes available, {@link fillInPlaceholderAndReturnWriteStatus}
   * must be called.
   */
  private static class FuturePut {
    /** Where the fingerprint should be written when it becomes available. */
    private final int offset;

    /**
     * A {@link PutOperation} that may have not yet started.
     *
     * <p>This arises when associated bytes could be mid-computation in a different thread. The
     * future completes when those bytes are known and its asynchronous write to the {@link
     * #fingerprintValueService} has started. {@link PutOperation#writeStatus} provides the status
     * of that write.
     */
    private final ListenableFuture<PutOperation> futurePut;

    private FuturePut(int offset, ListenableFuture<PutOperation> futurePut) {
      this.offset = offset;
      this.futurePut = futurePut;
    }

    private ListenableFuture<PutOperation> getFuturePut() {
      return futurePut;
    }

    /**
     * Copies the fingerprint into {@code bytes} at {@link #offset}.
     *
     * <p>The caller must ensure that {@link #futurePut} is complete before calling this method.
     */
    private ListenableFuture<Void> fillInPlaceholderAndReturnWriteStatus(byte[] bytes)
        throws SerializationException {
      PutOperation put = getDoneSerializationFuture(futurePut);
      put.fingerprint().copyTo(bytes, offset);
      return put.writeStatus();
    }
  }
}
