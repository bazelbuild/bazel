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

#include "src/tools/remote/src/main/cpp/testonly_output_service/bazel_output_service_impl.h"

#include <stdint.h>
#include <stdio.h>

#include <memory>

#include "src/tools/remote/src/main/cpp/testonly_output_service/memory.h"
#include "src/tools/remote/src/main/cpp/testonly_output_service/string.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

grpc::Status BazelOutputServiceImpl::Clean(
    grpc::ServerContext* context,
    const bazel_output_service::CleanRequest* request,
    bazel_output_service::CleanResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "");
}

grpc::Status BazelOutputServiceImpl::StartBuild(
    grpc::ServerContext* context,
    const bazel_output_service::StartBuildRequest* request,
    bazel_output_service::StartBuildResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "");
}

grpc::Status BazelOutputServiceImpl::StageArtifacts(
    grpc::ServerContext* context,
    const bazel_output_service::StageArtifactsRequest* request,
    bazel_output_service::StageArtifactsResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "");
}

grpc::Status BazelOutputServiceImpl::FinalizeArtifacts(
    grpc::ServerContext* context,
    const bazel_output_service::FinalizeArtifactsRequest* request,
    bazel_output_service::FinalizeArtifactsResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "");
}

grpc::Status BazelOutputServiceImpl::FinalizeBuild(
    grpc::ServerContext* context,
    const bazel_output_service::FinalizeBuildRequest* request,
    bazel_output_service::FinalizeBuildResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "");
}

grpc::Status BazelOutputServiceImpl::BatchStat(
    grpc::ServerContext* context,
    const bazel_output_service::BatchStatRequest* request,
    bazel_output_service::BatchStatResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "");
}

constexpr uint16_t kDefaultPort = 8080;

struct ParsedCommandLine {
  Str8 error;
  uint16_t port;
};

static ParsedCommandLine* ParseCommandLine(Arena* arena, int argc,
                                           char** argv) {
  TemporaryMemory scratch = BeginScratch(arena);
  ParsedCommandLine* result = PushArray(arena, ParsedCommandLine, 1);
  result->port = kDefaultPort;
  Str8 port_prefix = Str8FromCStr("--port=");
  for (int i = 1; i < argc; ++i) {
    Str8 arg = Str8FromCStr(argv[i]);
    if (StartsWithStr8(arg, port_prefix)) {
      Str8 port_str = PushSubStr8(scratch.arena, arg, port_prefix.len);
      ParsedUInt32 port = ParseUInt32(port_str);
      if (port.value) {
        result->port = port.value;
      } else {
        result->error = PushStr8F(arena, "Not a valid port: %s", port_str.ptr);
        break;
      }
    } else {
      result->error = PushStr8F(arena, "Unknown command line: %s", arg.ptr);
      break;
    }
  }
  EndScratch(scratch);
  return result;
}

int RunServer(int argc, char** argv) {
  int exit_code = 0;
  TemporaryMemory scratch = BeginScratch(0);
  ParsedCommandLine* command_line = ParseCommandLine(scratch.arena, argc, argv);
  if (IsEmptyStr8(command_line->error)) {
    BazelOutputServiceImpl service;

    Str8 address = PushStr8F(scratch.arena, "0.0.0.0:%d", command_line->port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort((char*)address.ptr,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    fprintf(stderr, "Server listening on port %d...\n", command_line->port);

    server->Wait();
  } else {
    fprintf(stderr, "%s\n", command_line->error.ptr);
    exit_code = 1;
  }
  EndScratch(scratch);
  return exit_code;
}
