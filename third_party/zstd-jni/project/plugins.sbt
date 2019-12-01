resolvers += Resolver.url("joprice-sbt-plugins", url("http://dl.bintray.com/content/joprice/sbt-plugins"))(Resolver.ivyStylePatterns)

addSbtPlugin("com.github.joprice" % "sbt-jni" % "0.2.0")
addSbtPlugin("com.typesafe.sbt" % "sbt-osgi" % "0.9.4")
addSbtPlugin("com.github.sbt" % "sbt-jacoco" % "3.1.0")
