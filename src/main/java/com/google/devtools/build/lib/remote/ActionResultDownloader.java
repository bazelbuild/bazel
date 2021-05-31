package com.google.devtools.build.lib.remote;

import build.bazel.remote.execution.v2.ActionResult;
import com.google.common.util.concurrent.ListenableFuture;

public interface ActionResultDownloader {
  public ListenableFuture<Void> downloadActionResult(ActionResult actionResult);
}
