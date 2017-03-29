// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceBlockingStub;
import com.google.devtools.build.lib.remote.CasServiceGrpc.CasServiceStub;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasDownloadTreeMetadataRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasLookupRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadBlobRequest;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataReply;
import com.google.devtools.build.lib.remote.RemoteProtocol.CasUploadTreeMetadataRequest;
import io.grpc.stub.StreamObserver;
import java.util.Iterator;

/**
 * An abstraction layer between the remote execution client and gRPC to support unit testing. This
 * interface covers the CAS RPC methods, see {@link CasServiceBlockingStub} and
 * {@link CasServiceStub}.
 */
public interface GrpcCasInterface {
  CasLookupReply lookup(CasLookupRequest request);
  CasUploadTreeMetadataReply uploadTreeMetadata(CasUploadTreeMetadataRequest request);
  CasDownloadTreeMetadataReply downloadTreeMetadata(CasDownloadTreeMetadataRequest request);
  Iterator<CasDownloadReply> downloadBlob(CasDownloadBlobRequest request);
  StreamObserver<CasUploadBlobRequest> uploadBlobAsync(
      StreamObserver<CasUploadBlobReply> responseObserver);
}
