package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.remote.RemoteLease.LeaseStore;
import com.google.devtools.build.lib.remote.common.MissingDigestsFinder.Intention;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.LeaseService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Collection;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.locks.ReentrantLock;
import javax.annotation.concurrent.GuardedBy;

public class RemoteLeaseService implements LeaseService {
  private final String buildRequestId;
  private final String commandId;
  private final boolean verboseFailures;
  private final Path cacheDirectory;
  private final EventHandler eventHandler;
  private final RemoteCache remoteCache;
  private final Duration remoteCacheAge;
  private final Duration remoteCacheRenewInterval;

  private volatile boolean shouldRenew = false;
  private Thread renewThread;

  private final ReentrantLock leaseLock = new ReentrantLock();

  // Use Map instead of Set because we need to get the reference to the Lease and check
  // `expireAtEpochMilli` when checking whether it is alive.
  @GuardedBy("leaseLock")
  private final TreeMap<Lease, Lease> leases =
      new TreeMap<>(
          (o1, o2) -> {
            if (o1.equals(o2)) {
              return 0;
            }
            return o1.expireAtEpochMilli < o2.expireAtEpochMilli ? -1 : 1;
          });

  @VisibleForTesting
  static class Lease {
    private static final Lease MIN = now(0);

    // This doesn't contribute to the equality of the lease but is used for sorting so we can
    // qucikly collect a set of leases that are expired.
    private final long expireAtEpochMilli;
    private final byte[] digest;
    private final long size;
    private final int locationIndex;

    private static Lease now(long now) {
      return new Lease(now, new byte[0], /* size= */ 0, /* locationIndex= */ 0);
    }

    static Lease create(long expireAtEpochMilli, RemoteFileArtifactValue metadata) {
      return new Lease(
          expireAtEpochMilli,
          metadata.getDigest(),
          metadata.getSize(),
          metadata.getLocationIndex());
    }

    @VisibleForTesting
    Lease(long expireAtEpochMilli, byte[] digest, long size, int locationIndex) {
      this.expireAtEpochMilli = expireAtEpochMilli;
      this.digest = digest;
      this.size = size;
      this.locationIndex = locationIndex;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Lease)) {
        return false;
      }
      Lease lease = (Lease) o;
      return size == lease.size
          && locationIndex == lease.locationIndex
          && Arrays.equals(digest, lease.digest);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(Arrays.hashCode(digest), size, locationIndex);
    }
  }

  public RemoteLeaseService(
      String buildRequestId,
      String commandId,
      boolean verboseFailures,
      Path cacheDirectory,
      EventHandler eventHandler,
      RemoteCache remoteCache,
      Duration remoteCacheAge,
      Duration remoteCacheRenewInterval) {
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.verboseFailures = verboseFailures;
    this.cacheDirectory = cacheDirectory;
    this.eventHandler = eventHandler;
    this.remoteCache = remoteCache;
    this.remoteCacheAge = remoteCacheAge;
    this.remoteCacheRenewInterval = remoteCacheRenewInterval;
  }

  public void startBuild() {
    checkState(renewThread == null, "renewThread should be null");

    readLeases();

    try {
      var now = Instant.now().toEpochMilli();
      renewLeases(now, getLeasesToRenew(now));
    } catch (InterruptedException | ExecutionException e) {
      eventHandler.handle(Event.warn(Utils.grpcAwareErrorMessage(e, verboseFailures)));
    }

    shouldRenew = true;
    renewThread = new Thread(this::renewThreadMain, "remote-lease-renew");
    renewThread.setDaemon(true);
    renewThread.start();
  }

  public void finalizeBuild() {
    checkState(renewThread != null, "renewThread shouldn't be null");

    shouldRenew = false;
    renewThread.interrupt();
    renewThread = null;

    writeLeases();
  }

  private void readLeases() {
    Path leasesPath = getLeasesPath();
    if (!leasesPath.exists()) {
      return;
    }

    try (var silentClosable = Profiler.instance().profile("read leases")) {
      var leaseStore = LeaseStore.parseFrom(leasesPath.getInputStream());
      leaseLock.lock();
      try {
        for (var persistedLease : leaseStore.getLeasesList()) {
          var lease =
              new Lease(
                  persistedLease.getExpireAtEpochMilli(),
                  persistedLease.getDigest().toByteArray(),
                  persistedLease.getSize(),
                  persistedLease.getLocationIndex());
          this.leases.put(lease, lease);
        }
      } finally {
        leaseLock.unlock();
      }
    } catch (IOException e) {
      eventHandler.handle(Event.warn(Utils.grpcAwareErrorMessage(e, verboseFailures)));
    }
  }

  private void writeLeases() {
    try (var silentClosable = Profiler.instance().profile("write leases")) {
      var leaseStore = LeaseStore.newBuilder();

      leaseLock.lock();
      try {
        for (var lease : this.leases.keySet()) {
          leaseStore.addLeases(
              RemoteLease.Lease.newBuilder()
                  .setExpireAtEpochMilli(lease.expireAtEpochMilli)
                  .setDigest(ByteString.copyFrom(lease.digest))
                  .setSize(lease.size)
                  .setLocationIndex(lease.locationIndex)
                  .build());
        }
      } finally {
        leaseLock.unlock();
      }

      Path leasesPath = getLeasesPath();
      try {
        checkNotNull(leasesPath.getParentDirectory()).createDirectoryAndParents();
        leaseStore.build().writeTo(leasesPath.getOutputStream());
      } catch (IOException e) {
        eventHandler.handle(Event.warn(Utils.grpcAwareErrorMessage(e, verboseFailures)));
      }
    }
  }

  private Path getLeasesPath() {
    return cacheDirectory.getRelative("lease_store");
  }

  public void add(RemoteFileArtifactValue metadata) {
    var now = Instant.now().toEpochMilli();
    add(Lease.create(now + remoteCacheAge.toMillis(), metadata));
  }

  @VisibleForTesting
  void add(Lease lease) {
    leaseLock.lock();
    try {
      leases.put(lease, lease);
    } finally {
      leaseLock.unlock();
    }
  }

  @VisibleForTesting
  boolean isAlive(Lease lease) {
    return isAlive(lease.digest, lease.size, lease.locationIndex);
  }

  @Override
  public boolean isAlive(byte[] digest, long size, int locationIndex) {
    var now = Instant.now().toEpochMilli();
    leaseLock.lock();
    try {
      var lease = leases.get(new Lease(/* expireAtEpochMilli */ 0, digest, size, locationIndex));
      if (lease == null) {
        return false;
      }
      return now < lease.expireAtEpochMilli;
    } finally {
      leaseLock.unlock();
    }
  }

  @VisibleForTesting
  void renewLeases(long now, Collection<Lease> leasesToRenew)
      throws InterruptedException, ExecutionException {
    if (leasesToRenew.isEmpty()) {
      return;
    }

    try (var silentCloseable = Profiler.instance().profile("renew leases")) {
      ImmutableSet<Lease> renewedLeases = doRenewLease(now, leasesToRenew);

      leaseLock.lock();
      try {
        for (var lease : renewedLeases) {
          if (lease.expireAtEpochMilli <= now) {
            leases.remove(lease);
          } else {
            leases.put(lease, lease);
          }
        }
      } finally {
        leaseLock.unlock();
      }
    }
  }

  private ImmutableSet<Lease> doRenewLease(long now, Collection<Lease> leases)
      throws InterruptedException, ExecutionException {
    var metadata =
        TracingMetadataUtils.buildMetadata(buildRequestId, commandId, "remote-lease-renew", null);
    var context = RemoteActionExecutionContext.create(metadata);

    Iterable<Digest> digestsToQuery =
        () ->
            leases.stream()
                .map(lease -> DigestUtil.buildDigest(lease.digest, lease.size))
                .iterator();
    // TODO: disk cache, combined cache?
    ImmutableSet<Digest> missingDigests =
        remoteCache.findMissingDigests(context, Intention.READ, digestsToQuery).get();

    var result = ImmutableSet.<Lease>builderWithExpectedSize(leases.size());
    for (var lease : leases) {
      long expireAtEpochMilli = 0;
      if (!missingDigests.contains(DigestUtil.buildDigest(lease.digest, lease.size))) {
        expireAtEpochMilli = now + remoteCacheAge.toMillis();
      }
      result.add(new Lease(expireAtEpochMilli, lease.digest, lease.size, lease.locationIndex));
    }
    return result.build();
  }

  private ImmutableSet<Lease> getLeasesToRenew(long now) {
    leaseLock.lock();
    try {
      return ImmutableSet.copyOf(leases.subMap(Lease.MIN, Lease.now(now)).keySet());
    } finally {
      leaseLock.unlock();
    }
  }

  private void renewThreadMain() {
    while (shouldRenew) {
      try {
        Thread.sleep(remoteCacheRenewInterval.toMillis());

        var now = Instant.now().toEpochMilli();
        renewLeases(now, getLeasesToRenew(now));
      } catch (ExecutionException e) {
        eventHandler.handle(Event.warn(Utils.grpcAwareErrorMessage(e, verboseFailures)));
      } catch (InterruptedException e) {
        return;
      }
    }
  }

  @VisibleForTesting
  ImmutableSet<Lease> getAllLeases() {
    leaseLock.lock();
    try {
      return ImmutableSet.copyOf(leases.keySet());
    } finally {
      leaseLock.unlock();
    }
  }

  @VisibleForTesting
  RemoteCache getRemoteCache() {
    return remoteCache;
  }
}
