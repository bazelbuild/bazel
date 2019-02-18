package com.google.devtools.build.lib.authentication.aws;

import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.AWSCredentialsProviderChain;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.AuthHeaderRequest;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

public class AwsAuthModule extends BlazeModule {

  private static final String SERVICE = "s3";
  private final AuthHeadersProviderDelegate delegate = new AuthHeadersProviderDelegate();

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder)
      throws AbruptExitException {
    super.serverInit(startupOptions, builder);
    builder.addAuthHeadersProvider("aws", delegate);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    delegate.setDelegate(null);

    final AwsAuthOptions opts = env.getOptions().getOptions(AwsAuthOptions.class);
    final RemoteOptions remoteOpts = env.getOptions().getOptions(RemoteOptions.class);
    if (remoteOpts == null || opts == null) {
      return;
    }

    final AwsRegion region = AwsRegion.determineRegion(opts.awsRegion, remoteOpts.remoteCache);
    if (region == null) {
      return;
    }

    final AWSCredentialsProvider credsProvider = newCredsProvider(opts);
    if (credsProvider != null) {
      this.delegate.setDelegate(new AwsV4AuthHeadersProvider(region, SERVICE, credsProvider, true));
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(final Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(AwsAuthOptions.class)
        : ImmutableList.of();
  }

  @Nullable
  private static AWSCredentialsProvider newCredsProvider(final AwsAuthOptions opts)
      throws AbruptExitException {
    final ImmutableList.Builder<AWSCredentialsProvider> creds = ImmutableList.builder();

    if (opts.awsAccessKeyId != null || opts.awsSecretAccessKey != null) {
      ensure(opts.awsAccessKeyId != null, "AWS Access key provided, but missing Secret Key");
      ensure(opts.awsSecretAccessKey != null, "AWS Secret key provided, but missing Access Key");

      final BasicAWSCredentials basicAWSCredentials = new BasicAWSCredentials(
          opts.awsAccessKeyId, opts.awsSecretAccessKey);
      creds.add(new AWSStaticCredentialsProvider(basicAWSCredentials));
    }

    if (opts.awsProfile != null) {
      creds.add(new ProfileCredentialsProvider(opts.awsProfile));
    }

    if (opts.useAwsDefaultCredentials) {
      creds.add(DefaultAWSCredentialsProviderChain.getInstance());
    }

    final List<AWSCredentialsProvider> providers = creds.build();
    return providers.isEmpty() ? null : new AWSCredentialsProviderChain(providers);
  }

  private static void ensure(final boolean condition, final String msg) throws AbruptExitException {
    if (!condition) {
      throw new AbruptExitException(msg, ExitCode.COMMAND_LINE_ERROR);
    }
  }

  private static class AuthHeadersProviderDelegate implements AuthHeadersProvider {

    private volatile AuthHeadersProvider delegate;

    public void setDelegate(AuthHeadersProvider delegate) {
      this.delegate = delegate;
    }

    @Override
    public String getType() {
      return delegate.getType();
    }

    @Override
    public Map<String, List<String>> getRequestHeaders(AuthHeaderRequest request) throws IOException {
      Preconditions.checkState(delegate != null, "delegate has not been initialized");
      return delegate.getRequestHeaders(request);
    }

    @Override
    public void refresh() throws IOException {
      Preconditions.checkState(delegate != null, "delegate has not been initialized");
      delegate.refresh();
    }

    @Override
    public boolean isEnabled() {
      return delegate != null && delegate.isEnabled();
    }
  }
}
