package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploaderFactory;
import io.grpc.Context;
import javax.annotation.Nullable;

public class ByteStreamBuildEventArtifactUploaderFactory implements
    BuildEventArtifactUploaderFactory {

  private final ByteStreamUploader uploader;
  private final String remoteServerName;
  private final Context ctx;
  private final @Nullable String remoteInstanceName;

  ByteStreamBuildEventArtifactUploaderFactory(
      ByteStreamUploader uploader, String remoteServerName, Context ctx,
      @Nullable String remoteInstanceName) {
    this.uploader = uploader;
    this.remoteServerName = remoteServerName;
    this.ctx = ctx;
    this.remoteInstanceName = remoteInstanceName;
  }

  @Override
  public BuildEventArtifactUploader create() {
    return new ByteStreamBuildEventArtifactUploader(uploader.retain(), remoteServerName, ctx,
        remoteInstanceName);
  }
}
