How to update these files:

1. Go to http://search.maven.org/
2. Search for g:"com.google.protobuf"
3. Download the "jar" link from protobuf-java.
4. Download all binaries from "protoc".
5. Strip version number from protoc files: for i in *.exe; do mv $i $(echo $i | sed s/3.0.0-alpha-3-//); done
6. Set executable bit: chmod +x *.exe
7. Update third_party/BUILD to point to the new jar file.
8. Done.

