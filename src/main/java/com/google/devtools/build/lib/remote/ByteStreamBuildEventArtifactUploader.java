package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport.TransportKind;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.remoteexecution.v1test.Digest;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

class ByteStreamBuildEventArtifactUploader implements BuildEventArtifactUploader {

  private final ByteStreamUploader uploader;
  private final String remoteInstanceName;
  private final String remoteServerName;

  ByteStreamBuildEventArtifactUploader(ByteStreamUploader uploader, String remoteServerName,
      @Nullable String remoteInstanceName) {
    this.uploader = Preconditions.checkNotNull(uploader);
    this.remoteServerName = Preconditions.checkNotNull(remoteServerName);
    this.remoteInstanceName = remoteInstanceName;
  }

  @Override
  public PathConverter upload(Set<Path> files) throws IOException, InterruptedException {
    List<Chunker> blobs = new ArrayList<>(files.size());
    Map<Path, Digest> pathToDigest = new HashMap<>(files.size());

    for (Path file : files) {
      Chunker chunker = new Chunker(file);
      pathToDigest.put(file, chunker.digest());
      blobs.add(chunker);
    }

    uploader.uploadBlobs(blobs);

    return (Path path) -> {
        Digest digest = pathToDigest.get(path);
        if (digest == null) {
          // It's a programming error to reference a file that has not been uploaded.
          throw new IllegalStateException();
        }

        if (Strings.isNullOrEmpty(remoteInstanceName)) {
          return String.format("bytestream://%s/blobs/%s/%d", remoteServerName, digest.getHash(),
              digest.getSizeBytes());
        } else {
          return String.format("bytestream://%s/%s/blobs/%s/%d", remoteServerName,
              remoteInstanceName, digest.getHash(), digest.getSizeBytes());
        }
    };
  }

  @Override
  public List<TransportKind> supportedTransports() {
    return Collections.singletonList(TransportKind.BES_GRPC);
  }
}
