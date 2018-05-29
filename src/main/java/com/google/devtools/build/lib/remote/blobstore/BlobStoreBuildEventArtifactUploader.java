package com.google.devtools.build.lib.remote.blobstore;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport.TransportKind;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.remoteexecution.v1test.Digest;
import java.io.InputStream;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import javax.annotation.concurrent.ThreadSafe;

@ThreadSafe
public class BlobStoreBuildEventArtifactUploader implements BuildEventArtifactUploader {

  private final SimpleBlobStore blobStore;
  private final List<TransportKind> supportedTransports;

  public BlobStoreBuildEventArtifactUploader(SimpleBlobStore blobStore,
      List<TransportKind> supportedTransports) {
    this.blobStore = blobStore;
    this.supportedTransports = supportedTransports;
  }

  @Override
  public PathConverter upload(Set<Path> files) throws IOException, InterruptedException {
    ImmutableMap.Builder<Path, String> builder = ImmutableMap.builder();
    for (Path file : files) {
      if (!file.isFile(Symlinks.FOLLOW)) {
        throw new IOException("We only support uploading of files.");
      }
      byte[] contentHash = DigestUtils.getDigestOrFail(file, file.getFileSize(Symlinks.FOLLOW));
      Digest digest = DigestUtil.buildDigest(contentHash, file.getFileSize());
      try (InputStream in = file.getInputStream()) {
        builder.put(file, blobStore.put(digest.getHash(), digest.getSizeBytes(), in).toString());
      }
    }

    ImmutableMap<Path, String> pathConversions = builder.build();
    return (Path p) -> {
      String conversion = pathConversions.get(p);
      if (conversion == null) {
        // It's a programming error to reference a file that has not been uploaded.
        throw new IllegalStateException();
      }
      return conversion;
    };
  }

  @Override
  public List<TransportKind> supportedTransports() {
    return supportedTransports;
  }
}
