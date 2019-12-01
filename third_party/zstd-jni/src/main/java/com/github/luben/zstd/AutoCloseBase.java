package com.github.luben.zstd;

import java.io.Closeable;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

abstract class AutoCloseBase implements Closeable {

    private static final AtomicIntegerFieldUpdater<AutoCloseBase> SHARED_LOCK_UPDATER =
            AtomicIntegerFieldUpdater.newUpdater(AutoCloseBase.class, "sharedLock");
    private static final int SHARED_LOCK_CLOSED = -1;

    private boolean finalize = true;

    private volatile int sharedLock;

    /**
     * Enable or disable class finalizers
     *
     * If finalizers are disabled the responsibility fir calling the `close` method is on the consumer.
     *
     * @param finalize, default `true` - finalizers are enabled
     */
    public void setFinalize(boolean finalize) {
        this.finalize = finalize;
    }

    void storeFence() {
        // volatile field write has a storeFence effect. Note: when updated to Java 9+, this method could be replaced
        // with VarHandle.storeStoreFence().
        sharedLock = 0;
    }

    /**
     * For private library usage only. This call must be paired with a try block with {@link #releaseSharedLock()} in
     * the finally block.
     */
    void acquireSharedLock() {
        while (true) {
            int sharedLock = this.sharedLock;
            if (sharedLock < 0) {
                throw new IllegalStateException("Closed");
            }
            if (sharedLock == Integer.MAX_VALUE) {
                throw new IllegalStateException("Shared lock overflow");
            }
            if (SHARED_LOCK_UPDATER.compareAndSet(this, sharedLock, sharedLock + 1)) {
                break;
            }
        }
    }

    void releaseSharedLock() {
        while (true) {
            int sharedLock = this.sharedLock;
            if (sharedLock < 0) {
                throw new IllegalStateException("Closed");
            }
            if (sharedLock == 0) {
                throw new IllegalStateException("Shared lock underflow");
            }
            if (SHARED_LOCK_UPDATER.compareAndSet(this, sharedLock, sharedLock - 1)) {
                break;
            }
        }
    }

    abstract void doClose();

    @Override
    public void close() {
        // Note: still should use synchronized in addition to sharedLock, because this class must support racy close(),
        // the second could happen through finalization or cleaning (when Cleaner is used). When updated to Java 9+,
        // synchronization block could be replaced with try { ... } finally { Reference.reachabilityFence(this); }
        synchronized (this) {
            if (sharedLock == SHARED_LOCK_CLOSED) {
                return;
            }
            if (!SHARED_LOCK_UPDATER.compareAndSet(this, 0, SHARED_LOCK_CLOSED)) {
                throw new IllegalStateException("Attempt to close while in use");
            }
            doClose();
        }
    }

    @Override
    protected void finalize() {
        if (finalize) {
            close();
        }
    }
}
