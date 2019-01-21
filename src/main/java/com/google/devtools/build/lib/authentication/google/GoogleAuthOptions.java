package com.google.devtools.build.lib.authentication.google;

import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

public class GoogleAuthOptions extends OptionsBase {
  @Option(
      name = "google_default_credentials",
      oldName = "auth_enabled",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Whether to use 'Google Application Default Credentials' for authentication."
          + " See https://cloud.google.com/docs/authentication for details. Disabled by default."
  )
  public boolean useGoogleDefaultCredentials;

  @Option(
      name = "google_auth_scopes",
      oldName = "auth_scope",
      defaultValue = "https://www.googleapis.com/auth/cloud-platform",
      converter = CommaSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A comma-separated list of Google Cloud authentication scopes."
  )
  public List<String> googleAuthScopes;

  @Option(
      name = "google_credentials",
      oldName = "auth_credentials",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specifies the file to get authentication credentials from. See "
              + "https://cloud.google.com/docs/authentication for details."
  )
  public String googleCredentials;
}
