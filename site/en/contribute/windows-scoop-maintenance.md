Project: /_project.yaml
Book: /_book.yaml

# Maintaining Bazel Scoop package on Windows

{% include "_buttons.html" %}

Note: The Scoop package is experimental. To provide feedback, go to
`@excitoon` in issue tracker.

## Prerequisites {:#prerequisites}

You need:

*    [Scoop package manager](https://scoop.sh/) installed
*    GitHub account in order to publish and create pull requests to
     [scoopinstaller/scoop-main](https://github.com/scoopinstaller/scoop-main){: .external}
     * [@excitoon](https://github.com/excitoon){: .external} currently maintains this
       unofficial package. Feel free to ask questions by
       [e-mail](mailto:vladimir.chebotarev@gmail.com) or
       [Telegram](http://telegram.me/excitoon){: .external}.

## Release process {:#release-process}

Scoop packages are very easy to maintain. Once you have the URL of released
Bazel, you need to make appropriate changes in
[this file](https://github.com/scoopinstaller/scoop-main/blob/master/bucket/bazel.json){: .external}:

- update version
- update dependencies if needed
- update URL
- update hash (`sha256` by default)

In your filesystem, `bazel.json` is located in the directory
`%UserProfile%/scoop/buckets/main/bucket` by default. This directory belongs to
your clone of a Git repository
[scoopinstaller/scoop-main](https://github.com/scoopinstaller/scoop-main){: .external}.

Test the result:

```
scoop uninstall bazel
scoop install bazel
bazel version
bazel something_else
```

The first time, make a fork of
[scoopinstaller/scoop-main](https://github.com/scoopinstaller/scoop-main){: .external} and
specify it as your own remote for `%UserProfile%/scoop/buckets/main`:

```
git remote add mine FORK_URL
```

Push your changes to your fork and create a pull request.
