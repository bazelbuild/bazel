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

package com.google.devtools.build.lib.authandtls;

import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.AWSCredentialsProviderChain;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.google.common.collect.ImmutableList;

import java.io.IOException;
import java.util.List;

/** Utility methods for using {@link AuthAndTLSOptions} with Amazon Web Services. */
public final class AwsAuthUtils {

  /**
   * Create a new {@link com.amazonaws.auth.AWSCredentialsProvider} object.
   *
   * @throws IOException in case the credentials can't be constructed.
   */
  public static AWSCredentialsProvider newCredentials(AuthAndTLSOptions options) throws IOException {
    final ImmutableList.Builder<AWSCredentialsProvider> creds = ImmutableList.builder();

    if (options.awsAccessKeyId != null && options.awsSecretAccessKey != null) {
      final BasicAWSCredentials basicAWSCredentials = new BasicAWSCredentials(options.awsAccessKeyId, options.awsSecretAccessKey);
      creds.add(new AWSStaticCredentialsProvider(basicAWSCredentials));
    }

    if (options.awsProfile != null) {
      creds.add(new ProfileCredentialsProvider(options.awsProfile));
    }

    if (options.useAwsDefaultCredentials) {
      creds.add(DefaultAWSCredentialsProviderChain.getInstance());
    }

    final List<AWSCredentialsProvider> providers = creds.build();
    return providers.isEmpty() ? null : new AWSCredentialsProviderChain(providers);
  }
}
