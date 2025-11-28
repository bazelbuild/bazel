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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.FAILURE_REPORTING_CALLBACK;
import static com.google.devtools.build.lib.skyframe.serialization.FutureHelpers.waitForSerializationFuture;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.aggregateWriteStatuses;
import static com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.sparselyAggregateWriteStatuses;

import com.github.luben.zstd.ZstdOutputStream;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.AggregateWriteStatusBuilder;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.SparseAggregateWriteStatusBuilder;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatuses.WriteStatus;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import javax.annotation.Nullable;

/**
 * A {@link SerializationContext} that supports both memoization and shared subobjects.
 *
 * <p>Sharing subobjects occurs by means of uploading them asynchronously to a {@link
 * #fingerprintValueService} store. The status of these uploads may be observed through {@link
 * #createFutureToBlockWritingOn} and {@link SerializationResult#getFutureToBlockWritesOn}.
 */
abstract class SharedValueSerializationContext extends MemoizingSerializationContext {

  /** Size of serialized shared value after which we will compress the node. */
  public static final int COMPRESSION_THRESHOLD_IN_BYTES = 1024;

  final FingerprintValueService fingerprintValueService;

  /**
   * Futures that represent writes to remote storage.
   *
   * <p>For consistency, the serialized bytes should not be published for other consumers until
   * these writes complete.
   */
  @Nullable // lazily initialized
  private ArrayList<WriteStatus> futuresToBlockWritingOn;

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
    return create(
        codecRegistry, dependencies, fingerprintValueService, /* profileCollector= */ null);
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
      @Nullable Object subject,
      @Nullable ProfileCollector profileCollector)
      throws SerializationException {
    SharedValueSerializationContext context =
        create(codecRegistry, dependencies, fingerprintValueService, profileCollector);
    byte[] bytes = context.serializeToBytes(subject);
    if (context.futurePuts == null) {
      return context.createImmediateResult(bytes, /* allowSparseFutureAggregation= */ false);
    }
    return waitForSerializationFuture(context.resolveFuturePutsAndCreateResult(bytes));
  }

  static ListenableFuture<SerializationResult<ByteString>> serializeToResultAsync(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService,
      @Nullable Object subject,
      @Nullable ProfileCollector profileCollector) {
    SharedValueSerializationContext context =
        create(codecRegistry, dependencies, fingerprintValueService, profileCollector);
    byte[] bytes;
    try {
      bytes = context.serializeToBytes(subject);
    } catch (SerializationException e) {
      return immediateFailedFuture(e);
    }
    if (context.futurePuts == null) {
      return immediateFuture(
          context.createImmediateResult(bytes, /* allowSparseFutureAggregation= */ true));
    }
    return context.resolveFuturePutsAndCreateResult(bytes);
  }

  /**
   * Serializes {@code subject} to bytes.
   *
   * <p>The returned {@code byte[]} may contain placeholder values, requiring resolution of {@link
   * futurePuts}.
   */
  private byte[] serializeToBytes(@Nullable Object subject) throws SerializationException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    try {
      serialize(subject, codedOut);
      codedOut.flush();
    } catch (IOException e) {
      throw new SerializationException("Failed to serialize: " + subject, e);
    }
    if (getProfileRecorder() != null) {
      getProfileRecorder().checkStackEmpty(subject);
    }
    return bytesOut.toByteArray();
  }

  @Override
  public final <T> void putSharedValue(
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
      ((PackedFingerprint) previous).writeTo(codedOut);
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
   *       of those recursive calls are still being processed in a different thread</em>: this
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
   * the associated fingerprint must be written to the output bytes at {@link FuturePut#offset}. to
   * update placeholders present in the serialized bytes.
   */
  private final <T> void putOwnedSharedValue(
      T child,
      DeferredObjectCodec<T> codec,
      CodedOutputStream codedOut,
      SettableFuture<PutOperation> putOperation)
      throws IOException, SerializationException {
    byte[] childBytes; // bytes of serialized child, maybe containing placeholders
    // This is initially populated with the `futuresToWriteBlockingOn` of the child.
    ArrayList<WriteStatus> childWriteStatuses;
    // Each entry of the following list corresponds to a `putSharedValue` call made by serialization
    // of `child`. Used to fill-in `childBytes` placeholders.
    ArrayList<FuturePut> childFuturePuts;
    { // Serializes `child` with a fresh context, populating the above variables.
      ByteArrayOutputStream childStream = new ByteArrayOutputStream();
      CodedOutputStream childCodedOut = CodedOutputStream.newInstance(childStream);
      SharedValueSerializationContext childContext = getFreshContext();

      ProfileRecorder childRecorder = childContext.getProfileRecorder();
      if (childRecorder == null) {
        codec.serialize(childContext, child, childCodedOut);
      } else {
        childRecorder.pushLocation(codec);
        codec.serialize(childContext, child, childCodedOut);
        childRecorder.recordBytesAndPopLocation(/* startBytes= */ 0, childCodedOut);
      }

      childCodedOut.flush();
      childBytes = childStream.toByteArray();
      childWriteStatuses =
          childContext.futuresToBlockWritingOn == null
              ? new ArrayList<>()
              : childContext.futuresToBlockWritingOn;
      childFuturePuts = childContext.futurePuts;
    }

    if (childFuturePuts == null) {
      childBytes = maybeCompressBytes(childBytes);
      // There are no deferred bytes so `childBytes` is complete. Starts the upload.
      PackedFingerprint fingerprint = fingerprintValueService.fingerprint(childBytes);
      fingerprint.writeTo(codedOut); // Writes only the fingerprint to the stream.
      childWriteStatuses.add(fingerprintValueService.put(fingerprint, childBytes));

      WriteStatus writeStatus = sparselyAggregateWriteStatuses(childWriteStatuses);
      putOperation.set(new PutOperation(fingerprint, writeStatus));
      addFutureToBlockWritingOn(writeStatus);
      return;
    }

    var upload =
        new UploadOnceFuturePutsResolve(
            fingerprintValueService, childWriteStatuses, childBytes, childFuturePuts);

    putOperation.setFuture(upload);
    recordFuturePut(upload, codedOut);
  }

  private static byte[] maybeCompressBytes(byte[] childBytes) {
    if (childBytes.length > COMPRESSION_THRESHOLD_IN_BYTES) {
      ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
      outputStream.write((byte) 1);
      try (ZstdOutputStream zstdOutputStream = new ZstdOutputStream(outputStream)) {
        zstdOutputStream.write(childBytes);
        zstdOutputStream.flush();
        return outputStream.toByteArray();
      } catch (IOException e) {
        BugReporter.defaultInstance().sendBugReport(e);
        // Falls back onto uncompressed data.
      }
    }
    byte[] newChildBytes = new byte[childBytes.length + 1];
    newChildBytes[0] = (byte) 0;
    System.arraycopy(childBytes, 0, newChildBytes, 1, childBytes.length);
    return newChildBytes;
  }

  private static final class UploadOnceFuturePutsResolve extends ResolveFuturePuts<PutOperation> {
    private final FingerprintValueService fingerprintValueService;

    private UploadOnceFuturePutsResolve(
        FingerprintValueService fingerprintValueService,
        List<WriteStatus> childWriteStatuses,
        byte[] childBytes,
        Collection<FuturePut> childFuturePuts) {
      super(childWriteStatuses, childBytes, childFuturePuts, ForkJoinPool.commonPool());
      this.fingerprintValueService = fingerprintValueService;
      signalSubclassReady();
    }

    @Override
    protected PutOperation getValue() {
      // All placeholders are filled-in. Starts the upload.
      byte[] maybeCompressedBytes = maybeCompressBytes(childBytes);
      PackedFingerprint fingerprint = fingerprintValueService.fingerprint(maybeCompressedBytes);
      childWriteStatuses.add(fingerprintValueService.put(fingerprint, maybeCompressedBytes));
      return new PutOperation(fingerprint, childWriteStatuses.build());
    }
  }

  @Override // to narrow the return type
  public abstract SharedValueSerializationContext getFreshContext();

  @Override
  public final void addFutureToBlockWritingOn(WriteStatus future) {
    if (futuresToBlockWritingOn == null) {
      futuresToBlockWritingOn = new ArrayList<>();
    }
    futuresToBlockWritingOn.add(future);
  }

  @Override
  @Nullable
  public final WriteStatus createFutureToBlockWritingOn() {
    if (futurePuts == null) {
      if (futuresToBlockWritingOn == null) {
        return null;
      }
      return aggregateWriteStatuses(futuresToBlockWritingOn);
    }

    var aggregate = new AggregateWriteStatusBuilder();
    if (futuresToBlockWritingOn != null) {
      aggregate.addAll(futuresToBlockWritingOn);
    }
    for (FuturePut futurePut : futurePuts) {
      aggregate.add(
          Futures.transformAsync(
              futurePut.getFuturePut(), PutOperation::writeStatus, directExecutor()));
    }
    return aggregate.build();
  }

  private SerializationResult<ByteString> createImmediateResult(
      byte[] bytes, boolean allowSparseFutureAggregation) {
    checkState(futurePuts == null, "There were futurePuts to resolve");
    var finishedBytes = ByteString.copyFrom(bytes);
    if (futuresToBlockWritingOn == null) {
      return SerializationResult.createWithoutFuture(finishedBytes);
    }
    return SerializationResult.create(
        finishedBytes,
        allowSparseFutureAggregation
            ? sparselyAggregateWriteStatuses(futuresToBlockWritingOn)
            : aggregateWriteStatuses(futuresToBlockWritingOn));
  }

  /**
   * Records a deferred fingerprint value to be written to the serialized bytes.
   *
   * <p>Adds the corresponding entry to {@link #futurePuts} and inserts placeholder bytes into
   * {@code codedOut}.
   */
  private final void recordFuturePut(
      ListenableFuture<PutOperation> futurePut, CodedOutputStream codedOut) throws IOException {
    if (futurePuts == null) {
      futurePuts = new ArrayList<>();
    }
    futurePuts.add(new FuturePut(codedOut.getTotalBytesWritten(), futurePut));
    // Adds a placeholder for the real fingerprint to be filled in after the future completes.
    fingerprintValueService.fingerprintPlaceholder().writeTo(codedOut);
  }

  /**
   * Completes any placeholders in {@code bytes} that may be pending {@link #futurePuts}.
   *
   * <p>Also adds write statuses entailed by {@link #futurePuts} to {@link
   * #futuresToBlockWritingOn}.
   *
   * @param bytes output bytes with placeholders reserved for {@link #futurePuts}
   * @return the fully resolved bytes once {@link #futurePuts} complete
   */
  private ListenableFuture<SerializationResult<ByteString>> resolveFuturePutsAndCreateResult(
      byte[] bytes) {
    return new WaitForChildByteCompletion(
        futuresToBlockWritingOn == null ? ImmutableList.of() : futuresToBlockWritingOn,
        bytes,
        futurePuts);
  }

  private static final class WaitForChildByteCompletion
      extends ResolveFuturePuts<SerializationResult<ByteString>> {
    private WaitForChildByteCompletion(
        List<WriteStatus> childWriteStatuses, byte[] childBytes, Collection<FuturePut> futurePuts) {
      super(childWriteStatuses, childBytes, futurePuts, directExecutor());
      signalSubclassReady();
    }

    @Override
    protected SerializationResult<ByteString> getValue() {
      return SerializationResult.create(
          ByteString.copyFrom(childBytes), childWriteStatuses.build());
    }
  }

  /**
   * A tuple consisting of a stream offset and a pending {@link PutOperation}.
   *
   * <p>When {@link PutOperation} becomes available, its {@link PutOperation#fingerprint} should be
   * copied to the output bytes starting at {@link #offset}.
   */
  private static final class FuturePut {
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
  }

  private abstract static class ResolveFuturePuts<T> extends QuiescingFuture<T> {
    final byte[] childBytes;
    final SparseAggregateWriteStatusBuilder childWriteStatuses =
        new SparseAggregateWriteStatusBuilder();

    private ResolveFuturePuts(
        List<WriteStatus> childWriteStatuses,
        byte[] childBytes,
        Collection<FuturePut> futurePuts,
        Executor executor) {
      super(executor);
      this.childWriteStatuses.addAll(childWriteStatuses);
      this.childBytes = childBytes;
      for (FuturePut futurePut : futurePuts) {
        increment();
        Futures.addCallback(
            futurePut.getFuturePut(), new PutStartedCallback(futurePut.offset), directExecutor());
      }
    }

    /**
     * Undoes the pre-increment, signaling readiness for {@link QuiescingFuture#getValue}.
     *
     * <p>Subclasses must call this method. This is a separate step because all the {@link
     * FuturePut}s could be done in the constructor and trigger {@link QuiescingFuture#getReady}
     * before the subclass is ready.
     */
    final void signalSubclassReady() {
      decrement();
    }

    private void acceptPutOperation(PutOperation put, int offset) {
      put.fingerprint().copyTo(childBytes, offset);
      childWriteStatuses.add(put.writeStatus());
      decrement();
    }

    private void notifyFailedPutOperation(Throwable t) {
      notifyException(t);
    }

    @Override
    protected final void doneWithError() {
      // All FuturePuts are done, but some of them had errors. Reports any write errors that would
      // otherwise be ignored by the caller due to the primary error.
      Futures.addCallback(childWriteStatuses.build(), FAILURE_REPORTING_CALLBACK, directExecutor());
    }

    private final class PutStartedCallback implements FutureCallback<PutOperation> {
      private final int offset;

      private PutStartedCallback(int offset) {
        this.offset = offset;
      }

      @Override
      public void onSuccess(PutOperation put) {
        acceptPutOperation(put, offset);
      }

      @Override
      public void onFailure(Throwable t) {
        notifyFailedPutOperation(t);
      }
    }
  }

  private static SharedValueSerializationContext create(
      ObjectCodecRegistry codecRegistry,
      ImmutableClassToInstanceMap<Object> dependencies,
      FingerprintValueService fingerprintValueService,
      @Nullable ProfileCollector profileCollector) {
    return profileCollector == null
        ? new SharedValueSerializationContextImpl(
            codecRegistry, dependencies, fingerprintValueService)
        : new SharedValueSerializationProfilingContext(
            codecRegistry, dependencies, fingerprintValueService, profileCollector);
  }

  private static final class SharedValueSerializationContextImpl
      extends SharedValueSerializationContext {
    private SharedValueSerializationContextImpl(
        ObjectCodecRegistry codecRegistry,
        ImmutableClassToInstanceMap<Object> dependencies,
        FingerprintValueService fingerprintValueService) {
      super(codecRegistry, dependencies, fingerprintValueService);
    }

    @Override
    public SharedValueSerializationContext getFreshContext() {
      return new SharedValueSerializationContextImpl(
          getCodecRegistry(), getDependencies(), fingerprintValueService);
    }

    @Override
    @Nullable
    public ProfileRecorder getProfileRecorder() {
      return null;
    }
  }

  private static final class SharedValueSerializationProfilingContext
      extends SharedValueSerializationContext {
    private final ProfileRecorder profileRecorder;

    private SharedValueSerializationProfilingContext(
        ObjectCodecRegistry codecRegistry,
        ImmutableClassToInstanceMap<Object> dependencies,
        FingerprintValueService fingerprintValueService,
        ProfileCollector profileCollector) {
      super(codecRegistry, dependencies, fingerprintValueService);
      this.profileRecorder = new ProfileRecorder(profileCollector);
    }

    @Override
    public SharedValueSerializationContext getFreshContext() {
      return new SharedValueSerializationProfilingContext(
          getCodecRegistry(),
          getDependencies(),
          fingerprintValueService,
          profileRecorder.getProfileCollector());
    }

    @Override
    public ProfileRecorder getProfileRecorder() {
      return profileRecorder;
    }
  }
}
