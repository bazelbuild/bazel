package com.google.devtools.build.lib.authentication.aws;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

enum AwsRegion {

  US_EAST_2("us-east-2", "s3.us-east-2.amazonaws.com", "s3.dualstack.us-east-2.amazonaws.com"),
  US_EAST_1("us-east-1", "s3.amazonaws.com", "s3.us-east-1.amazonaws.com", "s3-external-1.amazonaws.com", "s3.dualstack.us-east-1.amazonaws.com"),
  US_WEST_1("us-west-1", "s3.us-west-1.amazonaws.com", "s3-us-west-1.amazonaws.com", "s3.dualstack.us-west-1.amazonaws.com"),
  US_WEST_2("us-west-2", "s3.us-west-2.amazonaws.com", "s3-us-west-2.amazonaws.com", "s3.dualstack.us-west-2.amazonaws.com"),
  CA_CENTRAL_1("ca-central-1", "s3.ca-central-1.amazonaws.com", "s3-ca-central-1.amazonaws.com", "s3.dualstack.ca-central-1.amazonaws.com"),
  AP_SOUTH_1("ap-south-1", "s3.ap-south-1.amazonaws.com", "s3-ap-south-1.amazonaws.com", "s3.dualstack.ap-south-1.amazonaws.com"),
  AP_NORTHEAST_2("ap-northeast-2", "s3.ap-northeast-2.amazonaws.com", "s3-ap-northeast-2.amazonaws.com", "s3.dualstack.ap-northeast-2.amazonaws.com"),
  AP_NORTHEAST_3("ap-northeast-3", "s3.ap-northeast-3.amazonaws.com", "s3-ap-northeast-3.amazonaws.com", "s3.dualstack.ap-northeast-3.amazonaws.com"),
  AP_SOUTHEAST_1("ap-southeast-1", "s3.ap-southeast-1.amazonaws.com", "s3-ap-southeast-1.amazonaws.com", "s3.dualstack.ap-southeast-1.amazonaws.com"),
  AP_SOUTHEAST_2("ap-southeast-2", "s3.ap-southeast-2.amazonaws.com", "s3-ap-southeast-2.amazonaws.com", "s3.dualstack.ap-southeast-2.amazonaws.com"),
  AP_NORTHEAST_1("ap-northeast-1", "s3.ap-northeast-1.amazonaws.com", "s3-ap-northeast-1.amazonaws.com", "s3.dualstack.ap-northeast-1.amazonaws.com"),
  CN_NORTH_1("cn-north-1", "s3.cn-north-1.amazonaws.com.cn"),
  CN_NORTHWEST_1("cn-northwest-1", "s3.cn-northwest-1.amazonaws.com.cn"),
  EU_CENTRAL_1("eu-central-1", "s3.eu-central-1.amazonaws.com", "s3-eu-central-1.amazonaws.com", "s3.dualstack.eu-central-1.amazonaws.com"),
  EU_WEST_1("eu-west-1", "s3.eu-west-1.amazonaws.com", "s3-eu-west-1.amazonaws.com", "s3.dualstack.eu-west-1.amazonaws.com"),
  EU_WEST_2("eu-west-2", "s3.eu-west-2.amazonaws.com", "s3-eu-west-2.amazonaws.com", "s3.dualstack.eu-west-2.amazonaws.com"),
  EU_WEST_3("eu-west-3", "s3.eu-west-3.amazonaws.com", "s3-eu-west-3.amazonaws.com", "s3.dualstack.eu-west-3.amazonaws.com"),
  SA_EAST_1("sa-east-1", "s3.sa-east-1.amazonaws.com", "s3-sa-east-1.amazonaws.com", "s3.dualstack.sa-east-1.amazonaws.com"),;

  final String name;
  final String[] hosts;

  AwsRegion(final String name, final String... hosts) {
    this.name = name;
    this.hosts = hosts;
  }

  static final ImmutableMap<String, AwsRegion> HOST_LOOKUP;

  static {
    ImmutableMap.Builder<String, AwsRegion> builder = ImmutableMap.builder();
    for (final AwsRegion region : AwsRegion.values()) {
      for (final String host : region.hosts) {
        builder.put(host, region);
      }
    }
    HOST_LOOKUP = builder.build();
  }

  static AwsRegion forHost(final String host) {
    return HOST_LOOKUP.get(host);
  }

  static AwsRegion forUri(final URI uri) {
    return AwsRegion.forHost(uri.getHost());
  }

  public static Collection<String> names() {
    return Arrays.stream(AwsRegion.values())
        .map(region -> region.name)
        .collect(Collectors.toList());
  }

  public static AwsRegion fromName(final String s) {
    return Arrays.stream(AwsRegion.values())
        .filter(x -> x.name.equalsIgnoreCase(s))
        .findFirst()
        .orElse(null);
  }

  public static AwsRegion determineRegion(final String knownAwsRegion, final String httpCacheUri)
      throws AbruptExitException{

    if (!Strings.isNullOrEmpty(knownAwsRegion)) {
      final AwsRegion toReturn = AwsRegion.fromName(knownAwsRegion);
      if (toReturn == null) {
        final String msg = String.format("AWS region (%s) provided but is not a known (known=%s)",
            knownAwsRegion, Joiner.on(", ").join(AwsRegion.names()));
        throw new AbruptExitException(msg, ExitCode.COMMAND_LINE_ERROR);
      }
      return toReturn;
    }

    if (!Strings.isNullOrEmpty(httpCacheUri)) {
        return AwsRegion.forUri(URI.create(httpCacheUri));
    }

    return null;
  }
}
