package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.CacheCapabilities;
import com.google.common.collect.ImmutableSet;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.remote.RemoteLeaseService.Lease;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import java.time.Instant;
import java.util.concurrent.ExecutionException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteLeaseService} */
@RunWith(JUnit4.class)
public class RemoteLeaseServiceTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  @Test
  public void noLease() {
    var leaseService = createLeaseService();
    var metadata = createMetadata("content");

    assertThat(
            leaseService.isAlive(
                metadata.getDigest(), metadata.getSize(), metadata.getLocationIndex()))
        .isFalse();
  }

  @Test
  public void leaseIsAlive() {
    var leaseService = createLeaseService();
    var metadata = createMetadata("content");
    var now = Instant.now();
    var lease = Lease.create(now.plusSeconds(3600).toEpochMilli(), metadata);

    leaseService.add(lease);

    assertThat(leaseService.isAlive(lease)).isTrue();
  }

  @Test
  public void leaseIsExpired() {
    var leaseService = createLeaseService();
    var metadata = createMetadata("content");
    var now = Instant.now();
    var lease = Lease.create(now.minusSeconds(3600).toEpochMilli(), metadata);

    leaseService.add(lease);

    assertThat(leaseService.isAlive(lease)).isFalse();
  }

  @Test
  public void addsSameDigestDifferentExpiration_overrideExisting() {
    var leaseService = createLeaseService();
    var metadata = createMetadata("content");
    var now = Instant.now();
    var oldLease = Lease.create(now.minusSeconds(3600).toEpochMilli(), metadata);
    leaseService.add(oldLease);

    var newLease = Lease.create(now.plusSeconds(3600).toEpochMilli(), metadata);
    leaseService.add(newLease);

    assertThat(leaseService.isAlive(newLease)).isTrue();
    assertThat(leaseService.getAllLeases()).hasSize(1);
  }

  @Test
  public void renew_removeLeasesForMissingFiles() throws ExecutionException, InterruptedException {
    var leaseService = createLeaseService();
    var metadata = createMetadata("content");
    var now = Instant.now().toEpochMilli();
    var lease = Lease.create(0, metadata);
    leaseService.add(lease);
    assertThat(leaseService.getAllLeases()).hasSize(1);

    leaseService.renewLeases(now, ImmutableSet.of(lease));

    assertThat(leaseService.isAlive(lease)).isFalse();
    assertThat(leaseService.getAllLeases()).isEmpty();
  }

  @Test
  public void renew_extendsLeases() throws Exception {
    var leaseService = createLeaseService();
    var metadata = createFile(leaseService, "content");
    var now = Instant.now().toEpochMilli();
    var lease = Lease.create(0, metadata);
    leaseService.add(lease);

    leaseService.renewLeases(now, ImmutableSet.of(lease));

    assertThat(leaseService.isAlive(lease)).isTrue();
  }

  private RemoteLeaseService createLeaseService() {
    var fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var cacheDirectory = fileSystem.getPath("/tmp/action_cache");
    var eventHandler = new EventCollector();
    var remoteOptions = Options.getDefaults(RemoteOptions.class);
    var remoteCache =
        new RemoteCache(
            CacheCapabilities.newBuilder().build(),
            new InMemoryCacheClient(),
            remoteOptions,
            DIGEST_UTIL);
    return new RemoteLeaseService(
        "",
        "",
        false,
        cacheDirectory,
        eventHandler,
        remoteCache,
        remoteOptions.remoteCacheAge,
        remoteOptions.remoteCacheRenewInternal);
  }

  private RemoteFileArtifactValue createMetadata(String content) {
    var bytes = content.getBytes(UTF_8);
    var digest = DIGEST_UTIL.compute(bytes);
    return RemoteFileArtifactValue.create(
        HashCode.fromString(digest.getHash()).asBytes(), bytes.length, 0);
  }

  private RemoteFileArtifactValue createFile(RemoteLeaseService leaseService, String content)
      throws Exception {
    var remoteCache = leaseService.getRemoteCache();
    var metadata = createMetadata("content");
    var context =
        RemoteActionExecutionContext.create(
            TracingMetadataUtils.buildMetadata(
                "build-request-id", "command-id", "action-id", /* actionMetadata= */ null));
    remoteCache
        .uploadBlob(
            context,
            DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize()),
            ByteString.copyFrom("content".getBytes(UTF_8)))
        .get();
    return createMetadata(content);
  }
}
