// Copyright 2018 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.CapabilitiesGrpc;
import build.bazel.remote.execution.v2.CapabilitiesGrpc.CapabilitiesBlockingStub;
import build.bazel.remote.execution.v2.Compressor;
import build.bazel.remote.execution.v2.DigestFunction;
import build.bazel.remote.execution.v2.ExecutionCapabilities;
import build.bazel.remote.execution.v2.GetCapabilitiesRequest;
import build.bazel.remote.execution.v2.PriorityCapabilities;
import build.bazel.remote.execution.v2.PriorityCapabilities.PriorityRange;
import build.bazel.remote.execution.v2.RequestMetadata;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import io.grpc.CallCredentials;
import io.grpc.Channel;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** Fetches the ServerCapabilities of the remote execution/cache server. */
class RemoteServerCapabilities {

  @Nullable private final String instanceName;
  private final ReferenceCountedChannel channel;
  @Nullable private final CallCredentials callCredentials;
  private final long callTimeoutSecs;
  private final RemoteRetrier retrier;

  public RemoteServerCapabilities(
      @Nullable String instanceName,
      ReferenceCountedChannel channel,
      @Nullable CallCredentials callCredentials,
      long callTimeoutSecs,
      RemoteRetrier retrier) {
    this.instanceName = instanceName;
    this.channel = channel;
    this.callCredentials = callCredentials;
    this.callTimeoutSecs = callTimeoutSecs;
    this.retrier = retrier;
  }

  private CapabilitiesBlockingStub capabilitiesBlockingStub(
      RemoteActionExecutionContext context, Channel channel) {
    return CapabilitiesGrpc.newBlockingStub(channel)
        .withInterceptors(
            TracingMetadataUtils.attachMetadataInterceptor(context.getRequestMetadata()))
        .withCallCredentials(callCredentials)
        .withDeadlineAfter(callTimeoutSecs, TimeUnit.SECONDS);
  }

  public ServerCapabilities get(String buildRequestId, String commandId)
      throws IOException, InterruptedException {
    RequestMetadata metadata =
        TracingMetadataUtils.buildMetadata(buildRequestId, commandId, "capabilities", null);
    RemoteActionExecutionContext context = RemoteActionExecutionContext.create(metadata);
    GetCapabilitiesRequest request =
        instanceName == null
            ? GetCapabilitiesRequest.getDefaultInstance()
            : GetCapabilitiesRequest.newBuilder().setInstanceName(instanceName).build();
    ServerCapabilities caps =
        retrier.execute(
            () ->
                channel.withChannelBlocking(
                    channel ->
                        capabilitiesBlockingStub(context, channel).getCapabilities(request)));
    return caps;
  }

  static class ClientServerCompatibilityStatus {

    private final List<String> warnings;
    private final List<String> errors;

    private ClientServerCompatibilityStatus(List<String> warnings, List<String> errors) {
      this.warnings = warnings;
      this.errors = errors;
    }

    static class Builder {
      private final ImmutableList.Builder<String> warnings = ImmutableList.builder();
      private final ImmutableList.Builder<String> errors = ImmutableList.builder();

      public void addWarning(String message) {
        warnings.add(message);
      }

      public void addError(String message) {
        errors.add(message);
      }

      public ClientServerCompatibilityStatus build() {
        return new ClientServerCompatibilityStatus(warnings.build(), errors.build());
      }
    }

    public boolean isOk() {
      return warnings.isEmpty() && errors.isEmpty();
    }

    public List<String> getWarnings() {
      return warnings;
    }

    public List<String> getErrors() {
      return errors;
    }
  }

  private static void checkPriorityInRange(
      int priority,
      String optionName,
      PriorityCapabilities prCap,
      ClientServerCompatibilityStatus.Builder result) {
    if (priority != 0) {
      boolean found = false;
      StringBuilder rangeBuilder = new StringBuilder();
      for (PriorityRange pr : prCap.getPrioritiesList()) {
        rangeBuilder.append(String.format("%d-%d,", pr.getMinPriority(), pr.getMaxPriority()));
        if (pr.getMinPriority() <= priority && priority <= pr.getMaxPriority()) {
          found = true;
          break;
        }
      }
      if (!found) {
        String range = rangeBuilder.toString();
        if (!range.isEmpty()) {
          range = range.substring(0, range.length() - 1);
        }
        result.addError(
            String.format(
                "--%s %d is outside of server supported range %s.", optionName, priority, range));
      }
    }
  }

  public enum ServerCapabilitiesRequirement {
    NONE,
    CACHE,
    EXECUTION,
    EXECUTION_AND_CACHE,
  }

  /** Compare the remote server capabilities with those requested by current execution. */
  public static ClientServerCompatibilityStatus checkClientServerCompatibility(
      ServerCapabilities capabilities,
      RemoteOptions remoteOptions,
      DigestFunction.Value digestFunction,
      ServerCapabilitiesRequirement requirement) {
    ClientServerCompatibilityStatus.Builder result = new ClientServerCompatibilityStatus.Builder();
    boolean shouldCheckExecutionCapabilities =
        (requirement == ServerCapabilitiesRequirement.EXECUTION
            || requirement == ServerCapabilitiesRequirement.EXECUTION_AND_CACHE);
    boolean shouldCheckCacheCapabilities =
        (requirement == ServerCapabilitiesRequirement.CACHE
            || requirement == ServerCapabilitiesRequirement.EXECUTION_AND_CACHE);
    if (!(shouldCheckCacheCapabilities || shouldCheckExecutionCapabilities)) {
      return result.build();
    }

    // Check API version.
    ClientApiVersion.ServerSupportedStatus st =
        ClientApiVersion.current.checkServerSupportedVersions(capabilities);
    if (st.isUnsupported()) {
      result.addError(st.getMessage());
    }
    if (st.isDeprecated()) {
      result.addWarning(st.getMessage());
    }

    if (shouldCheckExecutionCapabilities) {
      // Check remote execution is enabled.
      ExecutionCapabilities execCap = capabilities.getExecutionCapabilities();
      if (!execCap.getExecEnabled()) {
        result.addError(
            "Remote execution is not supported by the remote server, or the current "
                + "account is not authorized to use remote execution.");
        return result.build(); // No point checking other execution fields.
      }

      // Check execution digest function. The protocol only later added
      // support for multiple digest functions for remote execution, so
      // check both the singular and repeated field.
      if (execCap.getDigestFunctionsList().isEmpty()
          && execCap.getDigestFunction() != DigestFunction.Value.UNKNOWN) {
        if (execCap.getDigestFunction() != digestFunction) {
          result.addError(
              String.format(
                  "Cannot use hash function %s with remote execution. "
                      + "Server supported function is %s",
                  digestFunction, execCap.getDigestFunction()));
        }
      } else if (!execCap.getDigestFunctionsList().contains(digestFunction)) {
        result.addError(
            String.format(
                "Cannot use hash function %s with remote execution. "
                    + "Server supported functions are: %s",
                digestFunction, execCap.getDigestFunctionsList()));
      }

      // Check execution priority is in the supported range.
      checkPriorityInRange(
          remoteOptions.remoteExecutionPriority,
          "remote_execution_priority",
          execCap.getExecutionPriorityCapabilities(),
          result);
    }

    if (shouldCheckCacheCapabilities) {
      // Check cache digest function.
      CacheCapabilities cacheCap = capabilities.getCacheCapabilities();
      if (!cacheCap.getDigestFunctionsList().contains(digestFunction)) {
        result.addError(
            String.format(
                "Cannot use hash function %s with remote cache. "
                    + "Server supported functions are: %s",
                digestFunction, cacheCap.getDigestFunctionsList()));
      }

      if (remoteOptions.remoteUploadLocalResults
          && !cacheCap.getActionCacheUpdateCapabilities().getUpdateEnabled()) {
        result.addWarning(
            "--remote_upload_local_results is set, but the remote cache does not support uploading "
                + "action results or the current account is not authorized to write local results "
                + "to the remote cache.");
      }

      if (remoteOptions.cacheCompression
          && !cacheCap.getSupportedCompressorsList().contains(Compressor.Value.ZSTD)) {
        result.addError(
            "--remote_cache_compression requested but remote does not support compression");
      }

      // Check result cache priority is in the supported range.
      checkPriorityInRange(
          remoteOptions.remoteResultCachePriority,
          "remote_result_cache_priority",
          cacheCap.getCachePriorityCapabilities(),
          result);
    }

    return result.build();
  }
}
