# aws-sdk-auth-lite

The following is an extraction of the important auth interfaces from the AWS SDK.
These interfaces are only those needed to establish credentials with AWS, and
include nothing else from the AWS SDK.

Major changes have been made to remove the dependencies on Joda-time, jackson
and other similar dependencies.
A few small changes have been made to use guava utilities over those provided
by the original AWS SDK.

This SDK is vastly stripped down from the original, and only covers the basics
required to read auth tokens in standard AWS ways

Patches are applied over released maven source jars and not git sources.

There is a _basic_ shell script `update.sh` that covers the patching process,
as well as by SDK versioned patch files in `patches-vs-$sdk_version`
directories.
