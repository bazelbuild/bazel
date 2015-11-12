new_http_archive(
  name = "groovy-sdk-artifact",
  url = "http://dl.bintray.com/groovy/maven/apache-groovy-binary-2.4.4.zip",
  sha256 = "a7cc1e5315a14ea38db1b2b9ce0792e35174161141a6a3e2ef49b7b2788c258c",
  build_file = "groovy.BUILD",
)
bind(
  name = "groovy-sdk",
  actual = "@groovy-sdk-artifact//:sdk",
)
bind(
  name = "groovy",
  actual = "@groovy-sdk-artifact//:groovy",
)

maven_jar(
  name = "junit-artifact",
  artifact = "junit:junit:4.12",
)
bind(
  name = "junit",
  actual = "@junit-artifact//jar",
)

maven_jar(
  name = "spock-artifact",
  artifact = "org.spockframework:spock-core:0.7-groovy-2.0",
)
bind(
  name = "spock",
  actual = "@spock-artifact//jar",
)
