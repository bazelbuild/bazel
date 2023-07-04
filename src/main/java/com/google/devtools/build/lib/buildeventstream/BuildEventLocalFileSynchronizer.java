// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildeventstream;

import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import java.util.Collection;

/**
 * An interface that is used to wait for downloads of remote outputs before sending out local BEP
 * events that reference these outputs.
 */
public interface BuildEventLocalFileSynchronizer {
  BuildEventLocalFileSynchronizer NO_OP = localFiles -> Futures.immediateVoidFuture();

  ListenableFuture<Void> waitForLocalFileDownloads(Collection<LocalFile> localFiles);
}
