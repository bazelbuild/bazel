// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.remote.worker;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import com.google.devtools.build.lib.remote.RemoteCache;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.disk.DiskCacheClient;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** A {@link RemoteCache} backed by an {@link DiskCacheClient}. */
class OnDiskBlobStoreCache extends RemoteCache {

  public OnDiskBlobStoreCache(RemoteOptions options, Path cacheDir, DigestUtil digestUtil) {
    super(
        new DiskCacheClient(cacheDir, /* verifyDownloads= */ true, digestUtil),
        options,
        digestUtil);
  }

  public boolean containsKey(Digest digest) {
    return ((DiskCacheClient) cacheProtocol).contains(digest);
  }

  @SuppressWarnings("ProtoParseWithRegistry")
  public void downloadTree(
      RemoteActionExecutionContext context, Digest rootDigest, Path rootLocation)
      throws IOException, InterruptedException {
    rootLocation.createDirectoryAndParents();
    Directory directory =
        Directory.parseFrom(Utils.getFromFuture(downloadBlob(context, rootDigest)));
    for (FileNode file : directory.getFilesList()) {
      Path dst = rootLocation.getRelative(file.getName());
      Utils.getFromFuture(downloadFile(context, dst, file.getDigest()));
      dst.setExecutable(file.getIsExecutable());
    }
    for (DirectoryNode child : directory.getDirectoriesList()) {
      downloadTree(context, child.getDigest(), rootLocation.getRelative(child.getName()));
    }
  }

  public void uploadActionResult(
      RemoteActionExecutionContext context, ActionKey actionKey, ActionResult actionResult)
      throws IOException, InterruptedException {
    cacheProtocol.uploadActionResult(context, actionKey, actionResult);
  }

  public DigestUtil getDigestUtil() {
    return digestUtil;
  }
}
