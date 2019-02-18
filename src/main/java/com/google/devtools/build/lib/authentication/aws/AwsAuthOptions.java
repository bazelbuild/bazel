package com.google.devtools.build.lib.authentication.aws;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;

public class AwsAuthOptions extends OptionsBase {

  @Option(
    name = "aws_default_credentials",
    defaultValue = "false",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Whether to use 'AWS Default Credentials' for authentication."
            + "See https://docs.aws.amazon.com/AWSJavaSDK/latest/javadoc/com/amazonaws/auth/DefaultAWSCredentialsProviderChain.html"
            + " for details. Disabled by default."
  )
  public boolean useAwsDefaultCredentials;

  @Option(
    name = "aws_access_key_id",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Use a specific AWS_ACCESS_KEY_ID for authentication"
  )
  public String awsAccessKeyId;

  @Option(
    name = "aws_secret_access_key",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Use a specific AWS_SECRET_ACCESS_KEY for authentication"
  )
  public String awsSecretAccessKey;

  @Option(
    name = "aws_profile",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Use a specific profile for credentials"
  )
  public String awsProfile;

  @Option(
      name = "aws_region",
      defaultValue = "null",
      category = "remote",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Override AWS region detection and force a specific bucket region"
  )
  public String awsRegion;


}
