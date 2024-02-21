// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.auto.value.AutoValue;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.protobuf.ByteString;

/** Tuple representing a {@link FingerprintValueStore#put} operation. */
@AutoValue
public abstract class PutOperation {
  public static PutOperation create(ByteString fingerprint, ListenableFuture<Void> writeStatus) {
    return new AutoValue_PutOperation(fingerprint, writeStatus);
  }

  /** Key used to store the value. */
  public abstract ByteString fingerprint();

  /** The result of storing the value in the {@link FingerprintValueStore}. */
  public abstract ListenableFuture<Void> writeStatus();
}
