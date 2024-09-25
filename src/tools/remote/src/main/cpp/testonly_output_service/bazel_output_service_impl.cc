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

#include <iostream>
#include <memory>
#include <string>

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

int RunServer(int argc, char** argv) {
  BazelOutputServiceImpl service;

  std::string server_address = "0.0.0.0:8080";

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cerr << "Server listening on " << server_address << std::endl;

  server->Wait();

  return 0;
}
