# Bazel Maven Config Generator

This directory contains the source code to a website that runs on
[Google Apps Script](https://script.google.com). It crawls Maven POM metadata to
generate `WORKSPACE` configs using [`java_import_external`](../java.bzl).

Please watch the [demo video](https://www.youtube.com/watch?v=xdMDuhJTKMI) on
YouTube.

## Features

- Defines all transitive relationships
- Resolves diamond conflicts by bumping versions
- Calculates SHA256 (slow due to Apps Script API issue)
- Documents `licenses` heuristically categorizes
- Heuristics for `neverlink` (provided) jars
- Source jars and optional dependencies
- Mirrors jars to Google Drive
- Adds iBiblio URLs if 200 OK

## Verbosity

The huge config is good because it makes builds deterministic, highly available,
minimal, scalable, and *fast* because Bazel won't need to BFS HTTP POMs each
build. Even Java projects with hundreds of transitive dependencies can expect
`bazel fetch //....` to take seconds.

You'll also see all the mysterious code from the Internet that you're running on
your machine. For example, you might discover Apache Commons Collections 3.2.1
on the classpath, in which case it's game over if anything in the JVM is
deserializing. So the verbosity might actually save you from ending up in the
same boat as Equifax. See
[Operation Rosehub](https://opensource.googleblog.com/2017/03/operation-rosehub.html)
to learn more.

## Imperfection

Another reason why the generated config is huge is because it only gets you 90%
there. You will need to make subtle adjustments after using the tool.

### Optional Dependencies

Optional dependencies require careful consideration while tuning. If you click
the "Include optional dependencies" checkbox, you may quickly find yourself
overwhelmed by just how many there are. However, if you turn it off, you may
discover at runtime that some of those optional jars weren't so optional after
all.

Take for example Guava. The tool generates this:

```py
# Without optional dependencies
java_import_external(
    name = "com_google_guava",
    licenses = ["notice"],  # The Apache Software License, Version 2.0
    jar_sha256 = "36a666e3b71ae7f0f0dca23654b67e086e6c93d192f60ba5dfd5519db6c288c8",
    jar_urls = [
        "http://maven.ibiblio.org/maven2/com/google/guava/guava/20.0/guava-20.0.jar",
        "http://repo1.maven.org/maven2/com/google/guava/guava/20.0/guava-20.0.jar",
    ],
)

# With optional dependencies
java_import_external(
    name = "com_google_guava",
    licenses = ["notice"],  # The Apache Software License, Version 2.0
    jar_sha256 = "36a666e3b71ae7f0f0dca23654b67e086e6c93d192f60ba5dfd5519db6c288c8",
    jar_urls = [
        "http://maven.ibiblio.org/maven2/com/google/guava/guava/20.0/guava-20.0.jar",
        "http://repo1.maven.org/maven2/com/google/guava/guava/20.0/guava-20.0.jar",
    ],
    deps = [
        "@com_google_code_findbugs_jsr305",
        "@com_google_errorprone_error_prone_annotations",
        "@com_google_j2objc_annotations",
        "@org_codehaus_mojo_animal_sniffer_annotations",
    ],
)
```

But experience on several teams has shown us the optimal config is this:

```py
java_import_external(
    name = "com_google_guava",
    licenses = ["notice"],  # The Apache Software License, Version 2.0
    jar_sha256 = "36a666e3b71ae7f0f0dca23654b67e086e6c93d192f60ba5dfd5519db6c288c8",
    jar_urls = [
        "http://maven.ibiblio.org/maven2/com/google/guava/guava/20.0/guava-20.0.jar",
        "http://repo1.maven.org/maven2/com/google/guava/guava/20.0/guava-20.0.jar",
    ],
    exports = [
        "@com_google_code_findbugs_jsr305",
        "@com_google_errorprone_error_prone_annotations",
    ],
)
```

Not including optional dependencies is a good starting point, but in practice,
your config is probably going to end up somewhere in-between.

### Exports

Tuning the output with the `exports` attribute can be very helpful when taken in
moderation. It's good for Guava in the example above, because it's nonobvious
when or why those annotations might be needed at runtime. However the canonical
example would be a dependency injection library (e.g. Dagger, Guice) exporting
`@javax_inject`, since nearly everything that depends on the DI library will
statically reference those classes too.

### Test Libraries

It's strongly recommended that you add `testonly = 1` to testing libraries, so
they'll be isolated from production code. The tool currently doesn't have this
feature.

### Licenses

The tool only looks at pom.xml files to find the license(s). This includes both the
artifact POM and any parents it might have. Every unique license it finds will
be documented. Categories merge towards restrictiveness.

If the tool can't figure out or categorize the license(s), it'll leave you a
TODO. The [java.bzl](../java.bzl) file talks a little bit about what the license
categories mean. You can also read the heuristics in [index.html](index.html).

Please also note that package authors sometimes put third party classes in their
jars without declaring those licenses. So it's a good idea to actually look at
what's inside each jar.

### Annotation Processors

Significant adjustments will be need to made to annotation processing libraries.
For example, here's how one would
[configure Google Auto and Dagger](https://gist.github.com/jart/5333824b94cd706499a7bfa1e086ee00).

### Never Link

There are many subtleties to the appropriate use of `neverlink = 1`. For
example, the App Engine jars required many such adjustments in the
[Nomulus WORKSPACE](https://github.com/google/nomulus/blob/master/java/google/registry/repositories.bzl)
configuration.

One common pattern is to have a jar not linked by default, but you still want to
be able to link it in certain situations. In that case, the
`generated_linkable_rule_name` attribute is your friend. For example:

```py
java_import_external(
    name = "com_google_appengine_remote_api",
    jar_sha256 = "6ea6dc3b529038ea6b37e855cd1cd7612f6640feaeb0eec842d4e6d85e1fd052",
    jar_urls = [
        "http://domain-registry-maven.storage.googleapis.com/repo1.maven.org/maven2/com/google/appengine/appengine-remote-api/1.9.48/appengine-remote-api-1.9.48.jar",
        "http://repo1.maven.org/maven2/com/google/appengine/appengine-remote-api/1.9.48/appengine-remote-api-1.9.48.jar",
    ],
    licenses = ["permissive"],  # Google App Engine Terms of Service: https://cloud.google.com/terms/
    neverlink = True,
    generated_linkable_rule_name = "link",
)
```

In this case, `@com_google_appengine_remote_api` won't link and
`@com_google_appengine_remote_api//:link` will link.

### Jumbo Jars

Avoid jumbo jars when possible. It's a good idea to actually look inside each
one to make sure the author didn't shade or schlep in third party classes. That
can be problematic with this tool, because then: a) builds can't be minimal; and
b) difficult to troubleshoot classpath collisions might occur.

The author of a jumbo jar will hopefully offer a nonobvious way to opt-out.  For
example, `com.google.javascript:closure-compiler:v20170910` is a jumbo jar, but
you can use `com.google.javascript:closure-compiler-unshaded:v20170910` instead
which is minimal and fully specified.
