# To get the checksum of the versioned documentation tree tarball, run the
# following command with the selected Bazel version:
#
# $ curl -s https://mirror.bazel.build/bazel_versioned_docs/jekyll-tree-0.20.0.tar | sha256sum | cut -d" " -f1
# bb79a63810bf1b0aa1f89bd3bbbeb4a547a30ab9af70c9be656cc6866f4b015b

DOC_VERSIONS = [
    {
        "version": "0.20.0",
        "sha256": "bb79a63810bf1b0aa1f89bd3bbbeb4a547a30ab9af70c9be656cc6866f4b015b",
    },
    {
        "version": "0.19.2",
        "sha256": "3c2d9f21ec2fd1c0b8a310f6eb6043027c838810cdfc2457d4346a0e5cdcaa7a",
    },
    {
        "version": "0.19.1",
        "sha256": "ec892c59ba18bb8de1f9ae2bde937db144e45f28d6d1c32a2cee847ee81b134d",
    },
    {
        "version": "0.18.1",
        "sha256": "98b77f48e37a50fc6f83100bf53f661e10732bb3ddbc226e02d0225cb7a9a7d8",
    },
    {
        "version": "0.17.2",
        "sha256": "13b35dd309a0d52f0a2518a1193f42729c75255f5fae40cea68e4d4224bfaa2e",
    },
    {
        "version": "0.17.1",
        "sha256": "02256ddd20eeaf70cf8fcfe9b2cdddd7be87aedd5848d549474fb0358e0031d3",
    },
]
