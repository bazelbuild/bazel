bind(
  name = "groovyc",
  actual = "@groovy-bin//:groovyc",
)
new_http_archive(
  name = "groovy-bin",
  url = "http://dl.bintray.com/groovy/maven/apache-groovy-binary-2.4.4.zip",
  sha256 = "a7cc1e5315a14ea38db1b2b9ce0792e35174161141a6a3e2ef49b7b2788c258c",
  build_file = "groovy.BUILD",
)

bind(
  name = "groovy",
  actual = "@groovy-jar//jar",
)
maven_jar(
  name = "groovy-jar",
  artifact = "org.codehaus.groovy:groovy-all:2.3.7",
)

bind (
  name = "junit",
  actual = "@junit-jar//jar",
)
maven_jar(
  name = "junit-jar",
  artifact = "junit:junit:4.12",
)
