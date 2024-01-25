
# Upgrading Jacoco version

Upgrade of Jacoco can be done in three steps. They are needed because part of
the code is not handled by copybara.

## 1st pull request

- Add prebuilt jars for new version to third_party/java/jacoco
- Add new version to third_party/java/jacoco/BUILD
- Update latest version in the same build file
- Check if asm library needs to be updated as well

This PR needs to be merged.

## 2nd pull request

- Update versions in the tools/jdk/BUILD.java_tools
- Update asm versions in third_party/BUILD

Or anywhere else outside of third_party.

This PR is handled by copybara.

## 3rd pull request

- Remove prebuilt jars of the old version
- Update third_party/java/jacoco/BUILD accordingly

This PR needs to be merged.

