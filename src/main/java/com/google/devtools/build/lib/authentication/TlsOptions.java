package com.google.devtools.build.lib.authentication;

import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;

public class TlsOptions extends OptionsBase {

  @Option(
      name = "tls_enabled",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specifies whether to use TLS for remote execution/caching and the build event service"
              + " (BES)."
  )
  public boolean tlsEnabled;

  @Option(
      name = "tls_certificate",
      defaultValue = "",
      converter = OptionsUtils.PathFragmentConverter.class,
      valueHelp = "<path>",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Specify the TLS client certificate to use."
  )
  public PathFragment tlsCertificate;

  @Option(
      name = "tls_authority_override",
      defaultValue = "null",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "TESTING ONLY! Can be used with a self-signed certificate to consider the specified "
              + "value a valid TLS authority."
  )
  public String tlsAuthorityOverride;
}
