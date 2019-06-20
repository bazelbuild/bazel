package com.google.devtools.build.lib.remote;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

public class CombinedGrpcAndSimpleBlobStoreCache extends AbstractRemoteActionCache {

  private GrpcRemoteCache grpcCache;
  private SimpleBlobStoreActionCache simpleBlobCache;

  public CombinedGrpcAndSimpleBlobStoreCache(
      GrpcRemoteCache grpcCache,
      SimpleBlobStoreActionCache simpleBlobCache
  ) {
    super(grpcCache.options, grpcCache.digestUtil);
    this.grpcCache = grpcCache;
    this.simpleBlobCache = simpleBlobCache;
  }

  @Override
  public void close() {
    simpleBlobCache.close();
    grpcCache.close();
  }

  @Override
  public void upload(
      ActionKey actionKey,
      Action action,
      Command command,
      Path execRoot,
      Collection<Path> files,
      FileOutErr outErr)
      throws ExecException, IOException, InterruptedException {
    simpleBlobCache.upload(actionKey, action, command, execRoot, files, outErr);
    grpcCache.upload(actionKey, action, command, execRoot, files, outErr);
  }

  @Override
  protected ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
    return Futures.catchingAsync(
        simpleBlobCache.downloadBlob(digest, out),
        Throwable.class,
        (e) -> grpcCache.downloadBlob(digest, out), // TODO populate simpleBlobCache somehow?
        directExecutor()
    );
  }

  @Override
  public ActionResult getCachedActionResult(ActionKey actionKey)
      throws IOException, InterruptedException {
    ActionResult result = simpleBlobCache.getCachedActionResult(actionKey);
    if (result == null) {
      result = grpcCache.getCachedActionResult(actionKey);
      // TODO figure out how to populate simpleBlobCache with the result
    }
    return result;
  }
}
