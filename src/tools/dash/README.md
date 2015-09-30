# A Dashboard for Bazel

This is a self-hosted dashboard for Bazel. In particular, this runs a server
that turns build results and logs into webpages.

## Running the server

Build and run the server:

```bash
$ bazel build //src/tools/dash:dash
$ bazel-bin/src/tools/dash
```

Once you see the log message `INFO: Dev App Server is now running`, you
can visit [http://localhost:8080] to see the main page (which should say "No
builds, yet!").

This builds a .war file that can be deployed to AppEngine (although this
doc assumes you'll run it locally).

_Note: as of this writing, there is no authentication, rate limiting, or other
protection for the dashboard. Anyone who can access the URL can read and write
data to it. You may want to specify the `--address` or `--host` option
(depending on AppEngine SDK version) when you run `dash` to bind the server to
an internal network address._

## Configuring Bazel to write results to the dashboard

You will need to tell Bazel where to send build results. Run `bazel` with the
`--use_dash` and `--dash_url=http://localhost:8080` flags, for
example:

```bash
$ bazel build --use_dash --dash_url=http://localhost:8080 //foo:bar
```

If you don't want to have to specify the flags for every build and test, add
the following lines to your .bazelrc (either in your home directory,
_~/.bazelrc_, or on a per-project basis):

```
build --use_dash
build --dash_url=http://localhost:8080
```

Then build results will be sent to the dashboard by default.  You can specify
`--use_dash=false` for a particular build if you don't want it sent.

## Basic security

Default security of the dashboard is to allow anyone to read and write to it.
Security can then be enforced using IP filtering. However, this is sometime
not sufficient and restricting who can write to the dashboard is generally
a good idea. Bazel supports that to some extent. A secret should be shared
between the Bazel's client and the dashboard. In the dashboard, the secret
can be specified using a `BAZEL_DASH_SECRET` environment variable. In Bazel,
you specify that secret by setting the `--dash_secret` flag which specify a
path to the file containing the secret.

Example to run it with the development server with secret:

```
BAZEL_DASH_SECRET=secret bazel-bin/src/tools/dash &
echo secret >/path/to/secret
bazel test --use_dash --dash_url=http://localhost:8080 --dash_secret=/path/to/secret //test/...
```

Please email the
[mailing list](https://groups.google.com/forum/#!forum/bazel-discuss)
with any questions or concerns.
