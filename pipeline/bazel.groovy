#!/usr/bin/env groovy

@Library('snaprrlib') _

import main.groovy.org.shared.KubeConfig

kubeconfig = new KubeConfig(steps: this, team: "apollo")

def uploadArtifacts() {
    artifactory.upload(
        pattern: "uploads/*",
        target:  "bazel/external/bazelbuild/bazel/",
        failNoOp: false,
    )
}

def build_type() {
    if (env.GERRIT_EVENT_TYPE) {
        if (GERRIT_EVENT_TYPE == 'patchset-created') {
            return 'patchset'
        } else if (GERRIT_EVENT_TYPE == 'change-merged') {
            return 'postmerge'
        }
    }
    return "unknown"
}

pipeline {
    agent none

    options {
        timestamps()
        sendSplunkConsoleLog()
    }

    stages {
        stage('Build Bazel') {
            failFast true
            parallel {
                stage('darwin-x86_64') {
                    agent {
                        kubernetes {
                            cloud kubeconfig.cluster()
                            yaml kubeconfig.podYaml(pod: 'apollo-sumac', podBuilderLabel: "bazel5")
                        }
                    }
                    steps {
                        deleteDir()
                        script {
                            checkout([
                                $class: 'GitSCM',
                                branches: scm.branches,
                                extensions: scm.extensions + [[$class: 'CleanCheckout']],
                                userRemoteConfigs: scm.userRemoteConfigs
                            ])
                            withCredentials([
                                usernamePassword(credentialsId: 'tpipeline.user.password',
                                    usernameVariable: 'OD_USER',
                                    passwordVariable: 'OD_PASSWORD'),
                                usernamePassword(credentialsId: 'deployment2.artifactory.api.key',
                                    passwordVariable: 'ARTIFACTORY_PASSWORD',
                                    usernameVariable: 'ARTIFACTORY_USERNAME')
                            ]) {
                            def build_type = build_type()
                            sshagent(['jenkins.user.rsa.key']) {
                                sh """#!/bin/bash
                                    set -xeo pipefail
                                    cd \${WORKSPACE}
                                    ./pipeline/bazel.sh ${build_type}
                                """
                                }
                            }
                        }
                    }
                    post {
                        always {
                            echo "Completed, creating archives"
                            uploadArtifacts()
                        }
                        cleanup {
                            deleteDir()
                        }
                    }
                }
                stage('darwin-arm64') {
                    agent { label "m1mac" }
                    steps {
                        deleteDir()
                        script {
                            checkout([
                                $class: 'GitSCM',
                                branches: scm.branches,
                                extensions: scm.extensions + [[$class: 'CleanCheckout']],
                                userRemoteConfigs: scm.userRemoteConfigs
                            ])
                            withCredentials([
                                usernamePassword(credentialsId: 'tpipeline.user.password',
                                    usernameVariable: 'OD_USER',
                                    passwordVariable: 'OD_PASSWORD'),
                                usernamePassword(credentialsId: 'deployment2.artifactory.api.key',
                                    passwordVariable: 'ARTIFACTORY_PASSWORD',
                                    usernameVariable: 'ARTIFACTORY_USERNAME')
                            ]) {
                            def build_type = build_type()
                            sshagent(['jenkins.user.rsa.key']) {
                                sh """#!/bin/bash
                                    set -xeo pipefail
                                    cd \${WORKSPACE}
                                    ./pipeline/bazel.sh ${build_type}
                                """
                                }
                            }
                        }
                    }
                    post {
                        always {
                            echo "Completed, uploading archives"
                            uploadArtifacts()
                        }
                        cleanup {
                            deleteDir()
                        }
                    }
                }
                stage('linux-x86_64') {
                    agent { label "autonomy_vm" }
                    steps {
                        deleteDir()
                        script {
                            checkout([
                                $class: 'GitSCM',
                                branches: scm.branches,
                                extensions: scm.extensions + [[$class: 'CleanCheckout']],
                                userRemoteConfigs: scm.userRemoteConfigs
                            ])
                            withCredentials([
                                usernamePassword(credentialsId: 'tpipeline.user.password',
                                    usernameVariable: 'OD_USER',
                                    passwordVariable: 'OD_PASSWORD'),
                                usernamePassword(credentialsId: 'deployment2.artifactory.api.key',
                                    passwordVariable: 'ARTIFACTORY_PASSWORD',
                                    usernameVariable: 'ARTIFACTORY_USERNAME')
                            ]) {
                            def build_type = build_type()
                            sshagent(['jenkins.user.rsa.key']) {
                                sh """#!/bin/bash
                                    set -xeo pipefail
                                    export LC_ALL=C.UTF-8
                                    export LANG=C.UTF-8
                                    snaprr container up --envall \
                                        -i build-essential \
                                        -i openjdk-11-jdk \
                                        -i python \
                                        -i unzip \
                                        -i zip
                                    snaprr container run -w \${WORKSPACE} -- ./pipeline/bazel.sh ${build_type}
                                """
                                }
                            }
                        }
                    }
                    post {
                        always {
                            echo "Completed, creating archives"
                            uploadArtifacts()
                        }
                        cleanup {
                            deleteDir()
                        }
                    }
                }
            }
        }
    }
}
