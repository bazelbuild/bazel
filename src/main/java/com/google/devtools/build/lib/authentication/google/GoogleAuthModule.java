package com.google.devtools.build.lib.authentication.google;

import com.google.auth.Credentials;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.ServerBuilder;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

public class GoogleAuthModule extends BlazeModule {

  private final AuthHeadersProviderDelegate delegate = new AuthHeadersProviderDelegate();

  @Override
  public void serverInit(OptionsParsingResult startupOptions, ServerBuilder builder)
      throws AbruptExitException {
    super.serverInit(startupOptions, builder);
    builder.addAuthHeadersProvider("google", delegate);
  }

  @Override
  public void beforeCommand(CommandEnvironment env) throws AbruptExitException {
    GoogleAuthOptions opts = env.getOptions().getOptions(GoogleAuthOptions.class);
    if (opts == null) {
      delegate.setDelegate(null);
      return;
    }

    Credentials credentials = newCredentials(opts);

    if (credentials != null) {
      delegate.setDelegate(new GoogleAuthHeadersProvider(credentials));
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(GoogleAuthOptions.class)
        : ImmutableList.of();
  }

  @Nullable
  private static Credentials newCredentials(GoogleAuthOptions opts) throws AbruptExitException {
    Credentials credentials = null;
    if (opts.googleCredentials != null) {
      // Credentials from a file
      try (InputStream authFile = new FileInputStream(opts.googleCredentials)) {
        credentials = newCredentials(authFile, opts.googleAuthScopes);
      } catch (FileNotFoundException e) {
        String message = String.format("Could not open google auth credentials file '%s'",
            opts.googleCredentials);
        throw new AbruptExitException(message, ExitCode.COMMAND_LINE_ERROR, e);
      } catch (IOException e) {
        throw new AbruptExitException("Failed to initialize google auth credentials.",
            ExitCode.COMMAND_LINE_ERROR, e);
      }
    } else if (opts.useGoogleDefaultCredentials) {
      try {
        credentials = newCredentials(null /* application default */, opts.googleAuthScopes);
      } catch (IOException e) {
        throw new AbruptExitException("Failed to initialize google auth credentials.",
            ExitCode.COMMAND_LINE_ERROR, e);
      }
    }
    return credentials;
  }

  private static Credentials newCredentials(@Nullable InputStream credentialsFile,
      List<String> authScopes) throws IOException {
    GoogleCredentials creds = credentialsFile == null
        ? GoogleCredentials.getApplicationDefault()
        : GoogleCredentials.fromStream(credentialsFile);
    if (!authScopes.isEmpty()) {
      creds = creds.createScoped(authScopes);
    }
    return creds;
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
    public Map<String, List<String>> getRequestHeaders(URI uri) throws IOException {
      if (delegate == null) {
        throw new IllegalStateException("delegate has not been initialized");
      }
      return delegate.getRequestHeaders(uri);
    }

    @Override
    public void refresh() throws IOException {
      if (delegate == null) {
        throw new IllegalStateException("delegate has not been initialized");
      }
      delegate.refresh();
    }

    @Override
    public boolean isEnabled() {
      return delegate != null && delegate.isEnabled();
    }
  }
}
